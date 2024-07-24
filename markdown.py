import re
import textwrap
import logging
import os
import shutil
import tempfile

from concurrent.futures import ThreadPoolExecutor
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from pathlib import Path


class MarkdownEditor:
    def __init__(
        self,
        example_selector,
        llm_model,
        input_dir,
        original_file_path,
        replace_with_correction=False,
        verbose=True,
        corrected_file_name_suffix="_corrected"
    ):
        self.example_selector = example_selector
        self.llm_model = llm_model
        self.input_dir = input_dir
        self.original_file_path = original_file_path
        self.replace_with_correction = replace_with_correction
        self.verbose = verbose
        self.corrected_file_name_suffix = corrected_file_name_suffix

    def __lint(self):
        if self.original_file_path is not None:
            dir = os.path.dirname(self.original_file_path)
            glob_pattern = os.path.basename(self.original_file_path)
        else:
            dir = self.input_dir
            glob_pattern = "**/*.md"

        directory_path = Path(dir)

        files = list(directory_path.glob(glob_pattern))

        for file in files:
            file_path_lint = file.stem + "_lint_" + ".md"

            shutil.copy(file, f"{file.parent}/{file_path_lint}")

            os.system(
                f"markdownlint -f {file.parent}/{file_path_lint} -q"
            )

    def __get_lint_glob(self):
        if self.original_file_path is not None:
            dir = os.path.dirname(self.original_file_path)
            filename = os.path.basename(self.original_file_path)
            glob = os.path.splitext(filename)[0] + "_lint_.md"
        else:
            dir = self.input_dir
            glob = "**/*_lint_.md"
        return dir, glob


    def __load_lint_files(self):
        dir, glob = self.__get_lint_glob()

        loader = DirectoryLoader(
            dir,
            glob=glob,
            show_progress=False,
            loader_cls=TextLoader,
        )

        data = loader.load()
        return data

    def _cleanup_tmp_lint(self):
        dir, glob = self.__get_lint_glob()

        directory_path = Path(dir)
        files = list(directory_path.glob(glob))

        for file in files:
            file.unlink()

    def _get_header_texts(self, docs):
        headers = [doc.metadata for doc in docs]
        freeze_headers = set([frozenset(x.items()) for x in headers])
        for i, head in enumerate(freeze_headers):
            text = [x[1] for x in head if x[0] in ["Header 1", "Header 2", "Header 3", "Header 4"]][0]
            new_doc = Document(page_content=text, metadata = head)
            docs.append(new_doc)
        return docs

    def process_markdown(self):
        data = self.__lint()
        data = self.__load_lint_files()

        header_text_splitter = MarkdownHeaderTextSplitter(return_each_line=True, strip_headers=True, headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")])

        split_docs = []

        for doc in data:
            doc.page_content = self.__remove_code_tables_comments(doc.page_content)
            sub_split_docs = header_text_splitter.split_text(doc.page_content)

            for sub_doc in sub_split_docs:
                sub_doc.metadata['source'] = doc.metadata['source']

            split_docs = split_docs + sub_split_docs

        text_splitter = MarkdownTextSplitter(chunk_size=2000, chunk_overlap=0)

        data = text_splitter.transform_documents(split_docs)
        data = self._get_header_texts(data)

        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(self.__process_chunk, doc) for doc in data]
            results = [future.result() for future in futures]

        # Reduce results
        corrections_by_file = {}
        for result in results:
            if result[2].metadata["source"] not in corrections_by_file:
                corrections_by_file[result[2].metadata["source"]] = []
            corrections_by_file[result[2].metadata["source"]].append(result[0:2])

        for original_file_path, corrections in corrections_by_file.items():
            with open(original_file_path, "r") as file:
                modified_contents = file.read()

            for original, correction in corrections:
                if original in modified_contents:
                    if correction.strip() != "No corrections required.":
                        modified_contents = modified_contents.replace(
                            original, correction
                        )
                else:
                    # Log or handle cases where the original is not found
                    logging.warn("'%s' not found in the document.", original)

            if self.replace_with_correction:
                new_file_path = original_file_path.split("_lint_")[0] + ".md"  # Override file content in git mode
            else:
                new_file_path = (
                    original_file_path.split("_lint_")[0].rsplit(".", 1)[0]
                    + self.corrected_file_name_suffix
                    + ".md"
                )

            with open(new_file_path, "w") as file:
                print("!!!!!! Writing " + new_file_path)
                file.write(modified_contents)

        # Cleanup
        self._cleanup_tmp_lint()

    def __process_chunk(self, chunk):
        original = chunk.page_content
        # print(original)
        example_prompt = PromptTemplate(
            input_variables=["text", "correction"],
            template=textwrap.dedent(
                """\
                Text:<startoftext>{text}</endoftext>

                Correction:<startofcorrection>{correction}</end>"""
            ),
        )

        examples_selected = self.example_selector.select_examples({"text": original})

        prompt = FewShotPromptTemplate(
            examples=examples_selected,
            # example_selector=
            example_prompt=example_prompt,
            prefix=textwrap.dedent(
                """\
                [[[Instruction]]]
                You are an advanced AI text editor tasked with enhancing the rightness of the content, with the following specific instructions:
                1. Fix any incorrect markdown syntax for lists, bold, italic, and code blocks, but do not add any headings.
                2. Ensure proper spacing and line breaks are used for readability.
                3. Correct any obvious spelling errors in the markdown text.
                4. Do NOT modify the content or syntax of any mathematical equations, even if they appear to be incorrect or use non-standard notation.
                5. Equations enclosed in single dollar signs ($...$) or double dollar signs ($$...$$) should be left exactly as they are.
                6. If you encounter any LaTeX or MathJax syntax within equation delimiters, do not alter it in any way.
                7. Preserve all variables, numbers, and symbols within equations.
                8. If there are code blocks, ensure they are properly formatted but do not change the code itself.
                9. If there is no correction to make, respond with "No corrections required."
                10. Preserve the original markdown formatting, including markdown links.
                
                After making the necessary corrections, respond exclusively with the refined text, maintaining the essence and structure of the original content.
                """                               
            ),
            suffix=textwrap.dedent(
                """\
                Text:<startoftext>{text}</endoftext>

                Correction:<startofcorrection>"""
            ),
            input_variables=["text"],
        )

        chain = prompt | self.llm_model
        response = chain.invoke(original)
        correction = response.split("</end>")[0].split("</endofcorrection>")[0].split("</startofcorrection>")[0]
        logging.log(logging.DEBUG, "Correction: %s", correction)
        return original, correction, chunk

    def __remove_code_tables_comments(self, markdown_text):
        # Remove code blocks
        no_code = re.sub(r"```.*?```", "", markdown_text, flags=re.DOTALL)

        # Remove tables
        no_tables = re.sub(
            r"\|.*?\n\|[-| :]*\|.*?\n(\|.*?\n)*", "", no_code, flags=re.DOTALL
        )

        # Some tables do not have | for first column
        no_tables = re.sub(
            r"\n.*?\n[-| :]*\|.*?\n(.*?\n)*", "\n", no_tables, flags=re.DOTALL
        )

        # Remove HTML-style comments
        no_comments = re.sub(r"<!--.*?-->", "", no_tables, flags=re.DOTALL)

        # Remove anything between --- sections
        no_dashdashdash = re.sub(r"---.*?---", "", no_comments, flags=re.DOTALL)

        return no_dashdashdash
