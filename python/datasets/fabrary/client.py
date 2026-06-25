"""Read-only client for fabrary.net's public AppSync GraphQL API.

Access path (no secrets needed — same one anonymous website visitors use):
  1. mint unauthenticated AWS creds from fabrary's Cognito Identity Pool
  2. SigV4-sign each request to the AppSync endpoint
  3. send browser-ish headers so the WAF in front of AppSync admits us

Public decks carry every card's `cardIdentifier`, which is the Talishar slug
with dashes (Talishar import does str_replace('-','_', identifier); see
talishar/APIs/AddFavoriteDeck.php). That makes conversion to the engine's
decks/*.json mechanical.
"""
from __future__ import annotations

import json
import time
import urllib.request
import urllib.error

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials

REGION = "us-east-2"
POOL_ID = "us-east-2:e50f3ed7-32ed-4b22-a05e-10b3e7e03fe0"
ENDPOINT = "https://42xrd23ihbd47fjvsrt27ufpfe.appsync-api.us-east-2.amazonaws.com/graphql"
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

LIST_BY_HERO = """query($heroIdentifier:String,$nextToken:String){
  getPublicDecksWithResultsByHero(heroIdentifier:$heroIdentifier,nextToken:$nextToken){
    nextToken
    decks { deckId name format heroIdentifier }
  }
}"""

GET_DECK = """query($deckId:ID!){
  getDeck(deckId:$deckId){
    deckId name format heroIdentifier
    hero { cardIdentifier name types }
    matchups { matchupId heroIdentifiers name }
    deckCards {
      cardIdentifier
      quantity
      sideboardQuantity
      matchupQuantities { matchupId quantity sideboardQuantity }
      card { cardIdentifier name types subtypes pitch }
    }
  }
}"""


class Fabrary:
    def __init__(self) -> None:
        self._creds: Credentials | None = None

    def _ensure_creds(self) -> Credentials:
        if self._creds is None:
            ci = boto3.client("cognito-identity", region_name=REGION)
            ident = ci.get_id(IdentityPoolId=POOL_ID)["IdentityId"]
            c = ci.get_credentials_for_identity(IdentityId=ident)["Credentials"]
            self._creds = Credentials(c["AccessKeyId"], c["SecretKey"], c["SessionToken"])
        return self._creds

    # Gentle default delay between requests so we don't trip the WAF rate rule.
    request_delay = 0.8

    def gql(self, query: str, variables: dict | None = None, retries: int = 4) -> dict:
        body = json.dumps({"query": query, "variables": variables or {}})
        time.sleep(self.request_delay)
        for attempt in range(retries):
            req = AWSRequest(method="POST", url=ENDPOINT, data=body,
                             headers={"Content-Type": "application/json"})
            SigV4Auth(self._ensure_creds(), "appsync", REGION).add_auth(req)
            headers = dict(req.headers)
            headers.update({"Origin": "https://fabrary.net",
                            "Referer": "https://fabrary.net/", "User-Agent": _UA})
            r = urllib.request.Request(ENDPOINT, data=body.encode(), headers=headers,
                                       method="POST")
            try:
                with urllib.request.urlopen(r, timeout=30) as resp:
                    out = json.loads(resp.read().decode())
                if out.get("errors"):
                    raise RuntimeError(out["errors"])
                return out["data"]
            except urllib.error.HTTPError as e:
                if e.code in (401, 403) and attempt < retries - 1:
                    self._creds = None  # refresh creds and retry
                    time.sleep(1.0 + attempt)
                    continue
                raise

    def decks_by_hero(self, hero: str, max_decks: int = 20) -> list[dict]:
        out: list[dict] = []
        token = None
        while True:
            data = self.gql(LIST_BY_HERO, {"heroIdentifier": hero, "nextToken": token})
            page = data["getPublicDecksWithResultsByHero"]
            out.extend(page.get("decks") or [])
            token = page.get("nextToken")
            if not token or len(out) >= max_decks:
                return out[:max_decks]

    def get_deck(self, deck_id: str) -> dict:
        return self.gql(GET_DECK, {"deckId": deck_id})["getDeck"]


if __name__ == "__main__":
    import sys
    fab = Fabrary()
    hero = sys.argv[1] if len(sys.argv) > 1 else "dorinthea"
    decks = fab.decks_by_hero(hero, max_decks=10)
    cc = [d for d in decks if d.get("format") == "Classic Constructed"]
    print(f"hero={hero}: {len(decks)} decks, {len(cc)} Classic Constructed")
    for d in decks:
        print(f"   [{d['format']:>18}] {d['deckId']}  {d['name']}")
    if cc:
        full = fab.get_deck(cc[0]["deckId"])
        print("\n--- sample CC deck:", full["name"], "| hero:", full["hero"]["cardIdentifier"])
        for dc in (full["deckCards"] or [])[:8]:
            c = dc.get("card") or {}
            print(f"   x{dc['quantity']}  id={dc['cardIdentifier']:<28} types={c.get('types')} pitch={c.get('pitch')}")
