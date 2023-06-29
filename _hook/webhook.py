# https://github.com/Leo-G/Flask-Github-Webhooks-Handler/blob/master/github-webhook.py

from flask import Flask, render_template, request, redirect, url_for, jsonify

# http://pymotw.com/2/hmac/
import hmac
import hashlib

# http://techarena51.com/index.php/how-to-install-python-3-and-flask-on-linux/
import subprocess
import os
from webhook_config import GITHUB_SECRET, DIR


app = Flask(__name__)


def verify_hmac_hash(data, signature):
    github_secret = GITHUB_SECRET
    mac = hmac.new(github_secret, msg=data, digestmod=hashlib.sha1)
    return hmac.compare_digest("sha1=" + mac.hexdigest(), signature)


# https://docs.github.com/en/webhooks-and-events/webhooks/securing-your-webhooks
def verify_signature(payload_body, secret_token, signature_header):
    """Verify that the payload was sent from GitHub by validating SHA256.

    Raise and return 403 if not authorized.

    Args:
        payload_body: original request body to verify (request.body())
        secret_token: GitHub app webhook token (WEBHOOK_SECRET)
        signature_header: header received from GitHub (x-hub-signature-256)
    """
    if not signature_header:
        raise HTTPException(
            status_code=403, detail="x-hub-signature-256 header is missing!"
        )
    hash_object = hmac.new(
        secret_token.encode("utf-8"), msg=payload_body, digestmod=hashlib.sha256
    )
    expected_signature = "sha256=" + hash_object.hexdigest()
    if not hmac.compare_digest(expected_signature, signature_header):
        raise HTTPException(status_code=403, detail="Request signatures didn't match!")


@app.route("/webhook/", methods=["POST"])
def github_payload():
    signature = request.headers.get("X-Hub-Signature")
    data = request.data
    if verify_hmac_hash(data, signature):
        # if verify_signature(data, token, signature):
        if request.headers.get("X-GitHub-Event") == "ping":
            return jsonify({"msg": "Ok"})
        if request.headers.get("X-GitHub-Event") == "push":
            payload = request.get_json()
            if payload["commits"][0]["distinct"] == True:
                try:
                    subprocess.Popen(["git", "pull"], cwd=DIR)
                    subprocess.Popen(["bundle", "exec", "jekyll", "build"], cwd=DIR)
                    return jsonify({"message": "Git pull successful"}), 200
                except subprocess.CalledProcessError as error:
                    return jsonify({"msg": str(error.output)})
            else:
                return jsonify({"msg": "nothing to commit"})

    else:
        return jsonify({"msg": "invalid hash"})


if __name__ == "__main__":
    # app.debug = True
    app.run(host="127.0.0.1", port=5001)
