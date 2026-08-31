/**
 * @fileoverview Lists the things on this page a person could act on right now.
 *
 * This runs inside the page, because the questions it answers are questions
 * only the browser can answer. Is this button actually on screen? Is it behind
 * the cookie dialog? What does a screen reader call it? None of that is in the
 * HTML source; it is in the layout the browser just computed, and it changes
 * on every scroll.
 *
 * Four filters decide what survives, in this order, cheapest first:
 *
 * 1. **Actionable.** A role a person can operate — a link, a button, a field,
 *    a checkbox — whether that role comes from the tag or from an explicit
 *    `role` attribute. A `<div role="tooltip">` is not actionable, and neither
 *    is an `<a>` with no `href`, which is an anchor rather than a link.
 * 2. **Enabled.** A disabled control is drawn but cannot be operated, and
 *    offering it invites the agent to spend a step pressing it.
 * 3. **On screen.** Not `visibility: hidden`, not fully transparent, and with
 *    at least a {@link MIN_VISIBLE_PX}-pixel square of itself inside the
 *    viewport. The viewport, not the document: the agent sees one screenful at
 *    a time, so what is scrolled away is not offered, and the way to reach it
 *    is to scroll. That is also what keeps the element list and the screenshot
 *    describing the same thing.
 * 4. **Reachable.** A click at one of five points inside its visible box lands
 *    on it, and not on something covering it. This is the consent-overlay
 *    case: the page beneath is fully laid out and perfectly visible to
 *    `getBoundingClientRect`, and every click on it goes to the overlay.
 *
 * What survives is numbered from 1 in document order. The numbers mean nothing
 * beyond this one observation — the same button is 7 now and 4 after a scroll,
 * which is why history records what an element *was* rather than only its
 * number.
 *
 * **Not traversed:** iframes and shadow roots. Out of scope for v0, and
 * silently so — an element inside either is simply not listed, so the agent
 * cannot refer to it and the validator never sees a reference to check.
 */

(options) => {
  const MIN_VISIBLE_PX = 2;
  // How far inside a corner to aim a hit test. Sampling the exact corner of a
  // rounded or bordered element lands just outside it.
  const HIT_INSET_PX = 4;

  // Cast wide, then narrow by role. Everything natively interactive is named
  // here, plus every element carrying any `role` at all -- the role is checked
  // below rather than enumerated as ~20 more attribute selectors.
  const CANDIDATE_SELECTOR = [
    "a[href]",
    "button",
    "input",
    "select",
    "textarea",
    "summary",
    '[contenteditable=""]',
    '[contenteditable="true"]',
    "[role]",
  ].join(",");

  // The ARIA roles this agent's action space can actually operate: something
  // to press, something to type into, something to choose from. Roles that
  // only describe structure (`navigation`, `article`, `tooltip`) are absent on
  // purpose -- listing them would fill the planner's prompt with things no
  // action applies to.
  const ACTIONABLE_ROLES = new Set([
    "button",
    "checkbox",
    "combobox",
    "link",
    "listbox",
    "menuitem",
    "menuitemcheckbox",
    "menuitemradio",
    "option",
    "radio",
    "searchbox",
    "slider",
    "spinbutton",
    "switch",
    "tab",
    "textbox",
    "treeitem",
  ]);

  // What each `<input type>` is, in role terms. An absent type, or one no
  // browser recognises, falls back to `textbox` -- which is what the browser
  // itself renders for an unknown type.
  const INPUT_ROLES = {
    button: "button",
    checkbox: "checkbox",
    file: "button",
    image: "button",
    number: "spinbutton",
    radio: "radio",
    range: "slider",
    reset: "button",
    search: "searchbox",
    submit: "button",
  };

  // Input types whose value is worth reporting as visible text. `password` is
  // deliberately absent: its value is a secret, and an observation is written
  // to the run's trace.
  const TEXT_VALUE_TYPES = new Set([
    "",
    "date",
    "datetime-local",
    "email",
    "month",
    "number",
    "search",
    "tel",
    "text",
    "time",
    "url",
    "week",
  ]);

  const viewportWidth = document.documentElement.clientWidth;
  const viewportHeight = document.documentElement.clientHeight;
  const textLimit = options.textLimit;

  /**
   * Reduces text to the single-line form the planner is shown.
   *
   * @param {?string} value Raw text, possibly null and possibly full of the
   *     newlines and indentation of the page source.
   * @return {string} The text with runs of whitespace collapsed to one space,
   *     trimmed at both ends.
   */
  function collapse(value) {
    return (value || "").replace(/\s+/g, " ").trim();
  }

  /**
   * Caps a string at the observation's per-field length.
   *
   * @param {string} value Collapsed text.
   * @return {string} `value` unchanged when it fits, otherwise its first
   *     characters followed by an ellipsis, so a reader can tell that what
   *     they are looking at was cut.
   */
  function truncate(value) {
    if (value.length <= textLimit) {
      return value;
    }
    return value.slice(0, textLimit - 1) + "…";
  }

  /**
   * Clamps a coordinate into the viewport.
   *
   * @param {number} value A CSS-pixel coordinate.
   * @param {number} limit The viewport's extent on that axis.
   * @return {number} The coordinate, moved inside `[0, limit - 1]`.
   *     `elementFromPoint` answers null for anything outside the viewport, so
   *     an unclamped sample point would read as "occluded".
   */
  function clamp(value, limit) {
    return Math.min(Math.max(value, 0), limit - 1);
  }

  /**
   * Names the ARIA role an element presents, explicit or implicit.
   *
   * An explicit `role` wins outright, including when it makes a natively
   * interactive element non-actionable: `<button role="presentation">` is a
   * button the author has declared is not one, and it is excluded.
   *
   * @param {!Element} element The candidate.
   * @return {string} A role token, or the empty string for an element with
   *     neither an explicit role nor an implicit one.
   */
  function roleOf(element) {
    const explicit = collapse(element.getAttribute("role")).toLowerCase();
    if (explicit) {
      // The attribute takes a list, most-preferred first; browsers honour the
      // first token they understand, and so does this.
      return explicit.split(" ")[0];
    }
    const tag = element.tagName.toLowerCase();
    if (tag === "a") {
      return "link";
    }
    if (tag === "button" || tag === "summary") {
      return "button";
    }
    if (tag === "textarea") {
      return "textbox";
    }
    if (tag === "select") {
      // A drop-down is a combobox; one showing several rows at once is a list.
      return element.multiple || element.size > 1 ? "listbox" : "combobox";
    }
    if (tag === "input") {
      const type = (element.getAttribute("type") || "").toLowerCase();
      return INPUT_ROLES[type] || "textbox";
    }
    return element.isContentEditable ? "textbox" : "";
  }

  /**
   * Reports whether an element is drawn but cannot be operated.
   *
   * @param {!Element} element The candidate.
   * @return {boolean} True for a disabled form control, for a control inside a
   *     disabled `<fieldset>` (which the DOM property already accounts for),
   *     and for anything marked `aria-disabled="true"`.
   */
  function isDisabled(element) {
    return element.disabled === true || element.getAttribute("aria-disabled") === "true";
  }

  /**
   * Reports whether an element has been withdrawn from the page's users.
   *
   * Both attributes are inherited by descendants in effect, so the test is on
   * the element's ancestry rather than on the element alone.
   *
   * @param {!Element} element The candidate.
   * @return {boolean} True if the element or an ancestor is `aria-hidden` or
   *     `inert`. `hidden`, and `display: none` generally, need no test here:
   *     such an element has no box, and the box test rejects it.
   */
  function isWithdrawn(element) {
    return element.closest('[aria-hidden="true"], [inert]') !== null;
  }

  const transparency = new Map();

  /**
   * Reports whether an element is invisible because of opacity -- its own, or
   * an ancestor's.
   *
   * Opacity behaves unlike most CSS here. A child does not inherit its
   * parent's opacity, but it is still painted through it, so a wrapper set to
   * `opacity: 0` hides everything inside it however opaque those children
   * claim to be. The only way to answer the question is therefore to walk up
   * the ancestors and check each one.
   *
   * On a page with hundreds of elements that walk covers the same ancestors
   * over and over, so every answer is kept in a map and the walk stops at the
   * first element already in it.
   *
   * @param {?Element} element The candidate, or null at the top of the walk.
   * @return {boolean} True if the element is painted at zero opacity.
   */
  function isTransparent(element) {
    if (element === null) {
      return false;
    }
    const cached = transparency.get(element);
    if (cached !== undefined) {
      return cached;
    }
    const opaque = parseFloat(getComputedStyle(element).opacity) !== 0;
    const result = !opaque || isTransparent(element.parentElement);
    transparency.set(element, result);
    return result;
  }

  /**
   * Measures how much of an element lies inside the viewport.
   *
   * @param {!Element} element The candidate.
   * @return {?{left: number, top: number, right: number, bottom: number,
   *     width: number, height: number}} The element's box clipped to the
   *     viewport, or null if it is hidden or too little of it is on screen.
   *     The box is clipped rather than whole because the clipped box is what
   *     the screenshot shows and what a click can land in -- the half of a
   *     button below the fold is not somewhere to aim.
   */
  function visibleBox(element) {
    // Unlike opacity, `visibility` is inherited, so the value the browser has
    // already computed accounts for the ancestors by itself: hidden if one of
    // them hid it, and visible again if this element set `visibility: visible`
    // to undo that. One check, and no walk.
    if (getComputedStyle(element).visibility !== "visible") {
      return null;
    }
    if (isTransparent(element)) {
      return null;
    }
    const rect = element.getBoundingClientRect();
    const left = Math.max(rect.left, 0);
    const top = Math.max(rect.top, 0);
    const right = Math.min(rect.right, viewportWidth);
    const bottom = Math.min(rect.bottom, viewportHeight);
    const width = right - left;
    const height = bottom - top;
    if (width < MIN_VISIBLE_PX || height < MIN_VISIBLE_PX) {
      return null;
    }
    return { left, top, right, bottom, width, height };
  }

  /**
   * Chooses the points at which to test whether an element can be clicked.
   *
   * The centre plus four inset corners. One point is not enough -- a link half
   * covered by a sticky header is still perfectly usable, and its centre may
   * be the covered half -- and five is enough to notice that a full-page
   * overlay covers all of it.
   *
   * @param {!{left: number, top: number, right: number, bottom: number,
   *     width: number, height: number}} box A clipped box.
   * @return {!Array<!Array<number>>} Five `[x, y]` pairs inside the viewport.
   */
  function hitPoints(box) {
    const inset = Math.min(HIT_INSET_PX, box.width / 4, box.height / 4);
    const midX = (box.left + box.right) / 2;
    const midY = (box.top + box.bottom) / 2;
    const nearLeft = box.left + inset;
    const nearRight = box.right - inset;
    const nearTop = box.top + inset;
    const nearBottom = box.bottom - inset;
    return [
      [midX, midY],
      [nearLeft, nearTop],
      [nearRight, nearTop],
      [nearLeft, nearBottom],
      [nearRight, nearBottom],
    ].map(([x, y]) => [clamp(x, viewportWidth), clamp(y, viewportHeight)]);
  }

  /**
   * Reports whether a click aimed at an element would actually reach it.
   *
   * Three outcomes count as reaching it. The hit is the element. The hit is
   * inside it -- the `<span>` inside a link, which is the common case, since
   * the innermost element under the cursor is what `elementFromPoint` returns.
   * Or the hit contains it, which happens when the element itself does not
   * take pointer events and its wrapper answers for it.
   *
   * That last case excludes `<body>` and `<html>`, which contain everything
   * and so would prove nothing: they are what is returned when the point is
   * over blank page, and treating that as a hit would make every invisible
   * element look reachable.
   *
   * @param {!Element} element The candidate.
   * @param {!{left: number, top: number, right: number, bottom: number,
   *     width: number, height: number}} box Its clipped box.
   * @return {boolean} True if any sample point lands on the element.
   */
  function isReachable(element, box) {
    const body = document.body;
    const root = document.documentElement;
    for (const [x, y] of hitPoints(box)) {
      const hit = document.elementFromPoint(x, y);
      if (hit === null) {
        continue;
      }
      if (hit === element || element.contains(hit)) {
        return true;
      }
      if (hit !== body && hit !== root && hit.contains(element)) {
        return true;
      }
    }
    return false;
  }

  /**
   * Joins the text of the elements an `aria-labelledby` points at.
   *
   * @param {!Element} element The candidate.
   * @return {string} The referenced text, collapsed, or the empty string if
   *     the attribute is absent or every id it names is missing. `textContent`
   *     backs up `innerText` because a label is often a hidden element, and
   *     `innerText` of something not rendered is empty.
   */
  function labelledByText(element) {
    const ids = collapse(element.getAttribute("aria-labelledby")).split(" ");
    const parts = [];
    for (const id of ids) {
      const node = id ? document.getElementById(id) : null;
      if (node !== null) {
        parts.push(collapse(node.innerText || node.textContent));
      }
    }
    return collapse(parts.join(" "));
  }

  /**
   * Reads the text of the `<label>` elements bound to a form control.
   *
   * @param {!Element} element An `<input>`, `<select>`, or `<textarea>`.
   * @return {string} The label text, or the empty string when the control has
   *     no label. A label can be attached two ways -- `<label for="...">`
   *     beside the field, or a `<label>` wrapped around it -- and the
   *     browser's own `labels` list already covers both, so neither form is a
   *     special case here.
   */
  function labelText(element) {
    const labels = element.labels;
    if (!labels || labels.length === 0) {
      return "";
    }
    return collapse(Array.from(labels, (label) => label.innerText || label.textContent).join(" "));
  }

  /**
   * Computes what a screen reader would call an element.
   *
   * There is a published algorithm for working this out, and it is long. What
   * is implemented here are the sources that carry real names on real pages --
   * `aria-labelledby`, `aria-label`, a `<label>`, a button's caption, an
   * image's alt text, a `title`, the element's own text -- checked in the
   * order the specification puts them in. The rarer cases it also covers, such
   * as assembling a name out of everything nested inside the element, are left
   * out. Where one of those would have applied, the element ends up named by
   * its own visible text instead, which is rarely much worse.
   *
   * The name matters more than the tag. It is what the planner reasons about
   * ("press Accept"), and it is what history records so that a step referring
   * to element 4 still means something once 4 has become 9.
   *
   * @param {!Element} element The candidate.
   * @param {string} role Its role, from {@link roleOf}.
   * @return {string} The accessible name, or the empty string for an element
   *     that genuinely has none -- an unlabelled icon button, say, which is
   *     worth showing to the planner as nameless rather than hiding.
   */
  function accessibleName(element, role) {
    const labelled = labelledByText(element);
    if (labelled) {
      return labelled;
    }
    const label = collapse(element.getAttribute("aria-label"));
    if (label) {
      return label;
    }

    const tag = element.tagName.toLowerCase();
    if (tag === "input" || tag === "select" || tag === "textarea") {
      const native = labelText(element);
      if (native) {
        return native;
      }
      const type = (element.getAttribute("type") || "").toLowerCase();
      if (type === "image") {
        const alt = collapse(element.getAttribute("alt"));
        if (alt) {
          return alt;
        }
      }
      if (role === "button") {
        // A push button's caption is its `value`, and browsers supply a
        // default caption when there is none -- so the name is never empty.
        const value = collapse(element.value);
        if (value) {
          return value;
        }
        if (type === "submit") {
          return "Submit";
        }
        if (type === "reset") {
          return "Reset";
        }
      }
      const title = collapse(element.getAttribute("title"));
      if (title) {
        return title;
      }
      // Last, and only last: a placeholder is a hint, not a label, and a field
      // named by one loses its name the moment something is typed into it.
      return collapse(element.getAttribute("placeholder"));
    }

    const text = collapse(element.innerText || element.textContent);
    if (text) {
      return text;
    }
    // An icon-only link or button: the image inside it carries the name.
    const image = element.querySelector("img[alt]");
    if (image !== null) {
      const alt = collapse(image.getAttribute("alt"));
      if (alt) {
        return alt;
      }
    }
    return collapse(element.getAttribute("title"));
  }

  /**
   * Reads what the element currently shows.
   *
   * Distinct from the name, and worth having alongside it: the name of a
   * search box stays "Search terms" while its text becomes what was typed, and
   * an agent that cannot see the difference cannot tell whether its own last
   * action took effect.
   *
   * @param {!Element} element The candidate.
   * @return {string} The rendered text, collapsed. Empty for a control that
   *     shows nothing of itself -- an untouched field, a checkbox, a file
   *     picker -- and empty for a password field, whose value is a secret and
   *     is never read.
   */
  function visibleText(element) {
    const tag = element.tagName.toLowerCase();
    if (tag === "textarea") {
      return collapse(element.value);
    }
    if (tag === "select") {
      const chosen = Array.from(element.selectedOptions, (option) => option.text);
      return collapse(chosen.join(", "));
    }
    if (tag === "input") {
      const type = (element.getAttribute("type") || "").toLowerCase();
      if (TEXT_VALUE_TYPES.has(type)) {
        return collapse(element.value);
      }
      // submit / button / reset: the value is the caption drawn on the button.
      if (type === "submit" || type === "button" || type === "reset") {
        return collapse(element.value);
      }
      return "";
    }
    return collapse(element.innerText);
  }

  const elements = [];
  // `querySelectorAll` answers in document order, which is what makes the
  // numbering below deterministic: same page, same numbers, every time.
  for (const element of document.querySelectorAll(CANDIDATE_SELECTOR)) {
    const role = roleOf(element);
    if (!ACTIONABLE_ROLES.has(role)) {
      continue;
    }
    if (isDisabled(element) || isWithdrawn(element)) {
      continue;
    }
    const box = visibleBox(element);
    if (box === null) {
      continue;
    }
    if (!isReachable(element, box)) {
      continue;
    }
    elements.push({
      id: elements.length + 1,
      tag: element.tagName.toLowerCase(),
      role: role,
      text: truncate(visibleText(element)),
      name: truncate(accessibleName(element, role)),
      // Whole pixels. The screenshot these numbers get drawn onto is itself
      // made of pixels, so fractions buy nothing -- and they would make two
      // observations of an unchanged page look different in the trace over a
      // difference far too small to see.
      box: {
        x: Math.round(box.left),
        y: Math.round(box.top),
        width: Math.round(box.width),
        height: Math.round(box.height),
      },
    });
  }
  return elements;
}
