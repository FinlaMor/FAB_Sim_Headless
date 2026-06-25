// Regenerate slug_index.json from the fabrary/cards npm package, in the
// package's NATIVE format (no transform — future-proof, tracks upstream):
//   key   = cardIdentifier (native, dashed; == fabrary heroIdentifier)
//   value = the card object verbatim (legalFormats keep spaces, etc.)
import { cards } from "@flesh-and-blood/cards";
import { writeFileSync } from "node:fs";

const by_slug = {};
for (const c of cards) by_slug[c.cardIdentifier] = c;
const out = new URL("../../slug_index.json", import.meta.url);
writeFileSync(out, JSON.stringify({ by_slug }, null, 2));
console.log("wrote", Object.keys(by_slug).length, "cards ->", out.pathname);
