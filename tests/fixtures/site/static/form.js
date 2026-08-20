/**
 * @fileoverview Carries earlier answers forward through the multi-step form.
 *
 * A static file server holds no session state, so each step re-submits
 * everything collected so far and the query string becomes the form's entire
 * memory. On load, every parameter already in the URL is copied into the form
 * as a hidden input, so submitting step two sends step one's answer along with
 * it. The script drives any form marked with a `data-carry` attribute, which
 * is why all three steps can share one file.
 */

const form = document.querySelector("form[data-carry]");

if (form) {
  const params = new URLSearchParams(window.location.search);
  for (const [name, value] of params) {
    // A field the user is filling in on this step owns its own value; only
    // answers from earlier steps need carrying.
    if (form.elements.namedItem(name)) {
      continue;
    }
    const carried = document.createElement("input");
    carried.type = "hidden";
    carried.name = name;
    carried.value = value;
    form.append(carried);
  }
}
