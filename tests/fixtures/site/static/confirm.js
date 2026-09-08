/**
 * @fileoverview Raises the browser's own dialogs, and records the answer.
 *
 * `confirm` returns true for OK and false for Cancel, and this page writes
 * which one it got into the status line and onto `document.body` as a
 * `data-dialog` attribute. That is the whole point of the page: the dialog
 * itself leaves no trace, so the *consequence* of the answer has to.
 */

const status = document.getElementById("dialog-status");

/**
 * Records what the page did, in both places a test might look.
 *
 * @param {string} outcome One word: `cancelled`, `kept`, or `notified`.
 * @param {string} message The sentence a reader sees.
 * @return {void}
 */
function record(outcome, message) {
  document.body.dataset.dialog = outcome;
  status.textContent = message;
}

document.getElementById("confirm-button").addEventListener("click", () => {
  if (window.confirm("Cancel reservation 4471? This cannot be undone.")) {
    record("cancelled", "Reservation 4471 has been cancelled.");
  } else {
    record("kept", "Reservation 4471 is still held until Friday.");
  }
});

document.getElementById("alert-button").addEventListener("click", () => {
  window.alert("The reading room is open Monday to Saturday, 9am to 7pm.");
  record("notified", "Opening hours were shown.");
});
