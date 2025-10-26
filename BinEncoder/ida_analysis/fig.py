import fitz


def remove_annotations(input_pdf_path, output_pdf_path):
    # 打开PDF文档
    doc = fitz.open(input_pdf_path)
    for page in doc:
        # 获取当前页的所有注释
        annots = page.annots()
        for annot in annots:
            # 删除注释
            page.delete_annot(annot)
    # 保存修改后的PDF
    doc.save(output_pdf_path)
    doc.close()


if __name__ == "__main__":
    input_pdf_path = r'C:\Users\tianh\Desktop\vul_fig1.pdf'
    output_pdf_path = r'C:\Users\tianh\Desktop\vul_fig1_no_annotations.pdf'
    remove_annotations(input_pdf_path, output_pdf_path)