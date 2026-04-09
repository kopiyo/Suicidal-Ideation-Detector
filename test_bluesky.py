"""
Run this to test Bluesky authentication before using the app.
    python test_bluesky.py

You need a Bluesky App Password:
  Bluesky -> Settings -> Privacy and Security -> App Passwords -> Add App Password
"""
import urllib.request, urllib.error, json

# ── Fill these in ────────────────────────────────────────────────────────────
YOUR_HANDLE   = "d-opiyo.bsky.social"   # your own Bluesky handle
YOUR_PASSWORD = "vaay-uwen-v5po-myy5"       # your App Password (not main password)
TARGET_HANDLE = "bsky.app"                  # the account you want to analyse
# ─────────────────────────────────────────────────────────────────────────────

print("Step 1 - Logging in...")
login_url = "https://bsky.social/xrpc/com.atproto.server.createSession"
payload   = json.dumps({"identifier": YOUR_HANDLE, "password": YOUR_PASSWORD}).encode()
req       = urllib.request.Request(
    login_url, data=payload,
    headers={"Content-Type": "application/json", "User-Agent": "MindGuard/3.0"},
    method="POST"
)
try:
    with urllib.request.urlopen(req, timeout=15) as r:
        data  = json.loads(r.read().decode())
        token = data["accessJwt"]
        print(f"SUCCESS - Logged in as {data.get('handle','')}")
except Exception as e:
    print(f"LOGIN FAILED: {e}")
    raise SystemExit

print(f"\nStep 2 - Resolving target handle: {TARGET_HANDLE}")
resolve_url = f"https://bsky.social/xrpc/com.atproto.identity.resolveHandle?handle={TARGET_HANDLE}"
req = urllib.request.Request(
    resolve_url,
    headers={"User-Agent": "MindGuard/3.0", "Authorization": f"Bearer {token}"}
)
try:
    with urllib.request.urlopen(req, timeout=15) as r:
        did = json.loads(r.read().decode())["did"]
        print(f"SUCCESS - DID: {did}")
except Exception as e:
    print(f"ERROR: {e}")
    raise SystemExit

print(f"\nStep 3 - Fetching posts for {TARGET_HANDLE}...")
feed_url = f"https://bsky.social/xrpc/app.bsky.feed.getAuthorFeed?actor={did}&limit=5"
req = urllib.request.Request(
    feed_url,
    headers={"User-Agent": "MindGuard/3.0", "Authorization": f"Bearer {token}"}
)
try:
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read().decode())
        feed = data.get("feed", [])
        print(f"SUCCESS - Got {len(feed)} posts")
        for item in feed[:3]:
            record = item.get("post", {}).get("record", {})
            print(f"  [{record.get('createdAt','')[:10]}] {record.get('text','')[:70]}")
except Exception as e:
    print(f"ERROR: {e}")
    raise SystemExit

print("\nAll steps passed. Bluesky authentication is working correctly.")
print("You can now use the Bluesky tab in MindGuard.")