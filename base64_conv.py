import base64
import json

# read the credentials file
with open('google-credentials.json', 'rb') as f:
    credentials_bytes = f.read()
    base64_credentials = base64.b64encode(credentials_bytes).decode('utf-8')
    print(base64_credentials)

    # save to file
    with open('credentials_base64.txt', 'w') as out_file:
        out_file.write(base64_credentials)