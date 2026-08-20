/* The consent dialog is an ordinary obstacle, not a special case: one of the
   two buttons has to be pressed, and which one was pressed stays legible in
   the page afterwards. */

const modal = document.getElementById("consent-modal");
const status = document.getElementById("consent-status");
const notices = document.getElementById("notices");
const restricted = document.getElementById("restricted-message");

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
