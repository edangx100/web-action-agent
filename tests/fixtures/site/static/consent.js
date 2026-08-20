/**
 * @fileoverview Handles the consent dialog on the member notices page.
 *
 * The dialog is an ordinary obstacle rather than a special case: its overlay
 * covers the page, so nothing underneath can be clicked until one of the two
 * buttons is pressed. Which button was pressed then stays legible in the page
 * afterwards — in the status line, in the content shown, and in a `data-consent`
 * attribute on the body — so a test can tell an accepted page from a declined
 * one without replaying the click.
 */

const modal = document.getElementById("consent-modal");
const status = document.getElementById("consent-status");
const notices = document.getElementById("notices");
const restricted = document.getElementById("restricted-message");

/**
 * Dismisses the dialog and reveals whichever content the answer allows.
 *
 * @param {boolean} accepted True if the reader accepted the data notice.
 * @return {void}
 */
function decide(accepted) {
  modal.hidden = true;
  document.body.dataset.consent = accepted ? "accepted" : "declined";
  status.textContent = accepted
    ? "Data notice accepted. Notices are shown below."
    : "Data notice declined.";
  notices.hidden = !accepted;
  restricted.hidden = accepted;
}

document.getElementById("accept-consent").addEventListener("click", () => decide(true));
document.getElementById("decline-consent").addEventListener("click", () => decide(false));
