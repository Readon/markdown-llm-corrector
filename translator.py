import os
import re
import textwrap

from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader

from markdown import MarkdownEditor


class MarkdownTranslator(MarkdownEditor):
    def __init__(
        self,
        llm_model,
        input_lang,
        target_lang,
        input_dir,
        original_file_path,
        replace_with_correction=False,
        verbose=True,
        name_suffix="_translated",        
        mid_file_suffix = "_corrected"
    ):
        super().__init__(
            self,
            llm_model=llm_model,
            input_dir=input_dir,
            original_file_path=original_file_path,
            replace_with_correction=replace_with_correction,
            verbose=verbose,
            out_file_suffix=name_suffix,
            mid_file_suffix=mid_file_suffix
        )
        self.input_lang = input_lang
        self.target_lang = target_lang
        self.return_each_line = True
    
    
    def _lint(self):
        pass

    def _get_lint_glob(self):
        if self.original_file_path is not None:
            dir = os.path.dirname(self.original_file_path)
            filename = os.path.basename(self.original_file_path)
            glob = filename
        else:
            dir = self.input_dir
            glob = "**/*.md"
        return dir, glob
    
    def _cleanup_tmp_lint(self):
        pass

    @staticmethod
    def _remove_tags(text:str) -> str:
        return re.sub(r'<startoftext>|</endoftext>|\[\[\[Output\]\]\]', '', text)
    
    def _construct_chain(self, original):
        input_lang = self.input_lang
        target_lang = self.target_lang
        template = f"""\
                [[[Instruction]]]
                You are a translation tool tasked with translating of the content, with the following specific instructions:
                1. You receive a text snippet from a file in the following format: Markdown. 
                2. The file is also written in the language:\n{input_lang}\n\n. As a translation tool, you will solely return the same string in {target_lang} without losing or amending the original formatting. 
                3. Your translations are accurate, aiming not to deviate from the original structure, content, writing style and tone.
                4. Do NOT modify the content or syntax of any mathematical equations, even if they appear to be incorrect or use non-standard notation.
                5. Equations enclosed in single dollar signs ($...$) or double dollar signs ($$...$$) should be left exactly as they are.
                6. If you encounter any LaTeX or MathJax syntax within equation delimiters, do not alter it in any way.
                7. Preserve all variables, numbers, and symbols within equations.
                8. If there are code blocks, ensure they are properly formatted but do not change the code itself.
                9. Preserve the original markdown formatting, including markdown links.
                10. Do not translate any number and acronym, and keep them in original style.
                11. Note for image should be translated.

                Text:<startoftext>{{text}}</endoftext>

                Translator:"""
        prompt = PromptTemplate(
            input_variables=["text"],
            template=textwrap.dedent(template),
        )

        chain = prompt | self.llm_model | StrOutputParser() | self._remove_tags
        return chain
