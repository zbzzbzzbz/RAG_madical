import os
import PyPDF2
import markdown
from bs4 import BeautifulSoup
import tiktoken
import json
import re
import csv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

enc = tiktoken.get_encoding("cl100k_base")
def tokens(text: str) -> int:
    return len(enc.encode(text))

class ReadFiles:
    """
    class to read files
    """
    
    def __init__(self, path: str) -> None:
        self._path = path
        self.file_list = self.get_files()   

    def get_files(self):
        # args：dir_path，目标文件夹路径
        file_list = []
        for filepath, dirnames, filenames in os.walk(self._path):
            # os.walk 函数将递归遍历指定文件夹
            for filename in filenames:
                # 通过后缀名判断文件类型是否满足要求
                if filename.endswith(".md"):
                    # 如果满足要求，将其绝对路径加入到结果列表
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".txt"):
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".pdf"):
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".csv"):
                    file_list.append(os.path.join(filepath, filename))
                else :
                    raise ValueError("Unsupported file type")
        return file_list

    def get_content(self, max_token_len: int = 600, cover_content: int = 150):
        docs = []
        # 读取文件内容
        for file in self.file_list:
            content = self.read_file_content(file)
            chunk_content = self.get_chunk(
                content, max_token_len=max_token_len, cover_content=cover_content)
            docs.extend(chunk_content)
        return docs

    @classmethod
    def read_file_content(cls, file_path: str):
        # 根据文件扩展名选择读取方法
        if file_path.endswith('.pdf'):
            return cls.read_pdf(file_path)
        elif file_path.endswith('.md'):
            return cls.read_markdown(file_path)
        elif file_path.endswith('.txt'):
            return cls.read_text(file_path)
        elif file_path.endswith('.csv'):
            return cls.read_csv(file_path)
        else:
            raise ValueError("Unsupported file type")

    @classmethod
    def read_pdf(cls, file_path: str):
        # 读取PDF文件，with 语句会自动管理文件的打开和关闭
        with open(file_path, 'rb') as file:
            # file 是文件对象， 不是文件路径
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
            return text    

    @classmethod
    def read_markdown(cls, file_path: str):
        # 读取Markdown文件
        with open(file_path, 'r', encoding='utf-8') as file:
            md_text = file.read()
            html_text = markdown.markdown(md_text)
            # 使用BeautifulSoup从HTML中提取纯文本
            soup = BeautifulSoup(html_text, 'html.parser')
            plain_text = soup.get_text()
            # 使用正则表达式移除网址链接
            text = re.sub(r'http\S+', '', plain_text) 
            return text
        
    @classmethod
    def read_text(cls, file_path: str):
        # 读取文本文件
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
        
    @classmethod
    def read_csv(cls, file_path: str):
        # 读取CSV文件
        text = ""
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)
            for row in tqdm(reader, desc="Reading .csv"):
                # 将每行数据用逗号连接成字符串，并添加换行符
                text += ','.join(row) + '\n'
            return text


    @classmethod
    def get_chunk(cls, text: str, max_token_len: int = 600, cover_content: int = 150):
        chunk_text = []
        lines = text.splitlines()  # 假设以换行符分割文本为行

        for line in lines:
            line_len = len(enc.encode(line))
            if line_len > max_token_len:
                # 如果单行长度就超过限制，则将其分割成多个块
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=max_token_len,  # chunk size (characters)
                    chunk_overlap=cover_content,  # chunk overlap (characters)
                    length_function = tokens,
                    # add_start_index=True,  # track index in original document
                )

                all_splits = text_splitter.split_text(line)
                for split in all_splits:
                    chunk_text.append(split)   
                
            else:
                chunk_text.append(line)

        return chunk_text


class Documents:
    """
        获取已分好类的json格式文档
    """
    def __init__(self, path: str = '') -> None:
        self.path = path
    
    def get_content(self):
        with open(self.path, mode='r', encoding='utf-8') as f:
            content = json.load(f)
        return content
