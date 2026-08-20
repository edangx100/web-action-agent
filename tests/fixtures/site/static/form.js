/* Carries earlier steps forward. A static server cannot hold session state, so
   each step re-submits everything collected so far as hidden inputs and the
   query string is the whole of the form's memory. */

const form = document.querySelector("form[data-carry]");

if (form) {
  const params = new URLSearchParams(window.location.search);
  for (const [name, value] of params) {
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
