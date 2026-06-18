import zipfile
import xml.etree.ElementTree as ET

def read_docx(path):
    try:
        z = zipfile.ZipFile(path)
        xml_content = z.read('word/document.xml')
        tree = ET.XML(xml_content)
        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        return '\n'.join([node.text for node in tree.findall('.//w:t', ns) if node.text])
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    t1 = read_docx('FT.docx')
    t2 = read_docx('HCFRI_Journal_Paper.docx')
    with open('docx_dump.txt', 'w', encoding='utf-8') as f:
        f.write("=== FT.docx ===\n")
        f.write(t1)
        f.write("\n\n=== HCFRI_Journal_Paper.docx ===\n")
        f.write(t2)
    print("Done writing docx_dump.txt")
