import segno
import sys

if len(sys.argv) != 2:
    print(f"Error: Usage python {sys.argv[0]} <url>")
    sys.exit()

url = sys.argv[1]
code = segno.make_qr(url)
code.save(
    "code.png",
    scale=50,
)

