/**
 * @fileoverview Opens the archive ledger in a second tab, from a script.
 *
 * The markup version of the same thing is the `target="_blank"` link on the
 * page. Both exist because a browser reports them the same way but a page
 * author reaches them differently, and an agent has to survive either.
 */

document.getElementById("popup-button").addEventListener("click", () => {
  window.open("ledger.html", "_blank");
});
