import gzip
content = "Lots of content here"
f = gzip.open('Onlyfinnaly.log.gz', 'wb')
f.write(content)
f.close()

import gzip
f=gzip.open('Onlyfinnaly.log.gz','rb')
file_content=f.read()
print(file_content)
f.close()