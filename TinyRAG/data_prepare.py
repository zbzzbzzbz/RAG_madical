import chardet

def wash_text(file_path: str):
    # 读取文本文件
    txt_docs = " "
    with open(file_path, 'r',encoding='utf-8') as file:
        txt_docs = file.read()
        placeholder = '|||'
        txt_docs = txt_docs.replace('\n\n', placeholder)
        # 步骤 2: 将单独的 \n 替换为空格
        txt_docs = txt_docs.replace('\n', '')
        # 步骤 3: 将占位符换回单个 \n
        txt_docs = txt_docs.replace(placeholder, '\n')
    # 将修改后的文本写回原文件
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(txt_docs)
    return txt_docs


def convert_to_utf8(input_file_path, output_file_path=None):
    # 以二进制模式打开文件
    with open(input_file_path, 'rb') as file:
        raw_data = file.read()
        # 检测文件编码
        result = chardet.detect(raw_data)
        encoding = result['encoding']
    # 如果没有指定输出文件路径，则覆盖原文件
    if output_file_path is None:
        output_file_path = input_file_path

    if (encoding == "GB2312") :
        encoding = "gbk"
    # 以检测到的编码读取文件内容
    with open(input_file_path, 'r', encoding=encoding) as file:
        content = file.read()

    # 以 UTF-8 编码写入文件
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(content)

    print(f"文件已从 {encoding} 编码转换为 UTF-8 编码，并保存到 {output_file_path}")




    
# 使用示例
input_file = 'data/man_query.csv'
convert_to_utf8(input_file)