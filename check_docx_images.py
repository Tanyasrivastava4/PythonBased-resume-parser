import sys
import docx
import zipfile

_NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "v": "urn:schemas-microsoft-com:vml",
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}

def check_images(docx_path):
    doc = docx.Document(docx_path)
    
    # Check header images
    for i, sec in enumerate(doc.sections):
        hdr = sec.header
        blips = hdr._element.findall(".//a:blip", _NS)
        v_images = hdr._element.findall(".//v:imagedata", _NS)
        print(f"Header {i}: found {len(blips)} DrawingML images, {len(v_images)} VML images")
        
    # Check body images
    blips = doc.element.body.findall(".//a:blip", _NS)
    v_images = doc.element.body.findall(".//v:imagedata", _NS)
    print(f"Body: found {len(blips)} DrawingML images, {len(v_images)} VML images")

    # Check zip media files inside .docx package
    with zipfile.ZipFile(docx_path) as z:
        media_files = [f for f in z.namelist() if f.startswith("word/media/")]
        print(f"Media files in docx zip ({len(media_files)}): {media_files}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_docx_images.py \"path/to/resume.docx\"")
        sys.exit(1)
    check_images(sys.argv[1])