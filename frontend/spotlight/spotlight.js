/* SPOTLIGHT: optional guided walkthrough of the Streamlit story.

   Streamlit Components v2 renderer. frontend/spotlight.py passes the step registry as
   `data`; this module validates it, adds the SPOTLIGHT button and runs the walkthrough.
   It only reads the rendered page and may select a story tab to show a target: it never
   calls the API and never changes model, database, prediction or training state. */

// Registry contract. A broken step stops the renderer with an error that names the step,
// so Streamlit shows the error in place of the button and the walkthrough never starts.
const PLAIN_FIELDS = ["target", "title", "keyword"];
const CODE_FIELDS = ["what", "role", "code", "decision"];

const PAD = 6;
const GAP = 12;
const EDGE = 12;
const MIN_WIDTH = 280;
// Streamlit's app header overlays the top 60 px of the main area: content under it is not visible.
const HEADER = 60;
const SCROLL_TOP = HEADER + 16;
// A newly selected tab lays out its content within a few frames; give up after about one second.
const MAX_WAIT_FRAMES = 60;

const CLEANUP_EVENT = "spotlight-cleanup";
const MISSING_TEXT = "This element is not rendered in the current view.";

function checkText(text, name, plain) {
  if (typeof text !== "string" || text.trim() === "") {
    throw new Error(name + ": empty field");
  }
  const backticks = text.split("`").length - 1;
  if (plain && backticks > 0) {
    throw new Error(name + ": backtick not allowed in this field");
  }
  if (backticks % 2 === 1) {
    throw new Error(name + ": unpaired backtick");
  }
}

function checkRegistry(data) {
  if (!data || typeof data.tab_root !== "string" || !Array.isArray(data.steps) || data.steps.length === 0) {
    throw new Error("Spotlight: data must contain tab_root and a non-empty steps list");
  }
  data.steps.forEach(function (step, stepIndex) {
    const name = "steps[" + stepIndex + "]";
    PLAIN_FIELDS.forEach(function (field) { checkText(step[field], name + "." + field, true); });
    CODE_FIELDS.forEach(function (field) { checkText(step[field], name + "." + field, false); });
    if ("tab" in step) checkText(step.tab, name + ".tab", true);
  });
  return data.steps;
}

function pad(number) {
  return String(number).padStart(2, "0");
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(value, max));
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

// Writes step text as DOM nodes, never as HTML: every odd segment between backticks
// becomes a <code> identifier, every even segment stays plain text.
function fill(node, text) {
  node.textContent = "";
  text.split("`").forEach(function (part, partIndex) {
    if (partIndex % 2 === 1) {
      node.appendChild(element("code", "", part));
    } else if (part) {
      node.appendChild(document.createTextNode(part));
    }
  });
}

// Nearest scrolling ancestor of a page element. In Streamlit this is the main section
// (or the sidebar content), not the window.
function scrollContainer(node) {
  for (let parent = node.parentElement; parent; parent = parent.parentElement) {
    const overflow = getComputedStyle(parent).overflowY;
    if (overflow === "auto" || overflow === "scroll") return parent;
  }
  return document.scrollingElement;
}

export default function ({ parentElement, data }) {
  // Streamlit runs the renderer again without calling the previous cleanup when the data
  // or the theme change. Clean up any earlier instance first, so there is always exactly
  // one button, one dialog and one set of listeners.
  Array.from(parentElement.children).forEach(function (child) {
    if (child.classList.contains("spotlight-root")) child.dispatchEvent(new Event(CLEANUP_EVENT));
  });

  const steps = checkRegistry(data);

  // The card numbers follow the tab numbers: the n-th step with a tab belongs to tab n
  // (spotlight.py checks that the steps follow the tabs one to one). The overview step
  // has no number.
  const tabStepCount = steps.filter(function (step) { return step.tab; }).length;
  function tabNumber(stepIndex) {
    return steps.slice(0, stepIndex + 1).filter(function (step) { return step.tab; }).length;
  }

  const root = element("div", "spotlight-root");
  const start = element("button", "spotlight-start", "SPOTLIGHT");
  start.type = "button";
  start.setAttribute("aria-label", "Start the Spotlight walkthrough");

  const dialog = element("dialog", "spotlight-dialog");
  dialog.setAttribute("aria-labelledby", "spotlight-title");
  const hole = element("div", "spotlight-hole");
  const popover = element("section", "spotlight-popover");
  const head = element("div", "spotlight-head");
  const count = element("span", "spotlight-count");
  const close = element("button", "spotlight-close", "×");
  close.type = "button";
  close.setAttribute("aria-label", "Close");
  close.title = "Close";
  head.append(count, close);

  const body = element("div", "spotlight-body");
  body.setAttribute("aria-live", "polite");
  const title = element("h2", "spotlight-title");
  title.id = "spotlight-title";
  const list = element("dl", "spotlight-sections");
  const what = element("dd", "spotlight-what");
  const role = element("dd", "spotlight-role");
  const code = element("dd", "spotlight-code");
  const decision = element("dd", "spotlight-decision");
  list.append(
    element("dt", "", "Purpose"), what,
    element("dt", "", "In this project"), role,
    element("dt", "", "Source"), code,
    element("dt", "", "What to notice"), decision
  );
  const missing = element("p", "spotlight-missing", MISSING_TEXT);
  missing.hidden = true;
  body.append(title, list, missing);

  const actions = element("div", "spotlight-actions");
  const back = element("button", "spotlight-back", "Back");
  back.type = "button";
  const next = element("button", "spotlight-next", "Next");
  next.type = "button";
  actions.append(back, next);

  popover.append(head, body, actions);
  dialog.append(hole, popover);
  root.append(start, dialog);
  parentElement.appendChild(root);

  let index = 0;
  let frame = 0;
  let waitFrame = 0;
  let listening = false;
  const resizeObserver = new ResizeObserver(schedule);

  function storyTabs() {
    const tabRoot = document.querySelector(data.tab_root);
    return tabRoot ? Array.from(tabRoot.querySelectorAll('[role="tab"]')) : [];
  }

  function selectedTabElement() {
    return storyTabs().find(function (item) { return item.getAttribute("aria-selected") === "true"; }) || null;
  }

  function selectedTab() {
    const tab = selectedTabElement();
    return tab ? tab.textContent.trim() : null;
  }

  // Selects a story tab by its label. Presentation state only: with on_change="ignore"
  // Streamlit does not rerun the script on a tab switch. Returns true when the tab changed.
  function selectTab(label) {
    const tab = storyTabs().find(function (item) { return item.textContent.trim() === label; });
    if (!tab || tab.getAttribute("aria-selected") === "true") return false;
    tab.click();
    return true;
  }

  // A target is missing when it is not in the page, has no size (an inactive tab) or lies
  // beside the viewport (a collapsed sidebar), where no scrolling can bring it into view.
  function targetElement() {
    const node = document.querySelector(steps[index].target);
    if (!node) return null;
    const rect = node.getBoundingClientRect();
    if (!rect.width && !rect.height) return null;
    if (rect.right <= 0 || rect.left >= document.documentElement.clientWidth) return null;
    return node;
  }

  // app.py zooms the page (CSS zoom on its main block, which holds this component).
  // getBoundingClientRect() and the viewport size are screen pixels, while the dialog lays
  // out in zoomed pixels: screen pixels divided by this factor.
  function zoomFactor() {
    const width = dialog.offsetWidth;
    return width ? dialog.getBoundingClientRect().width / width : 1;
  }

  // Positions hole and popover from the target's live geometry: beside at full width,
  // below, above, beside at reduced width, else over the target's bottom-right corner,
  // always clamped to the viewport. Returns
  // false when the target is not fully in view or the popover had to overlap it.
  function place() {
    frame = 0;
    const node = targetElement();
    const zoom = zoomFactor();
    const viewWidth = document.documentElement.clientWidth / zoom;
    const viewHeight = window.innerHeight / zoom;
    hole.hidden = !node;
    missing.hidden = Boolean(node);
    dialog.classList.toggle("spotlight-dim", !node);
    popover.style.width = "";
    popover.style.maxHeight = "";
    let width = popover.offsetWidth;
    let height = popover.offsetHeight;

    if (!node) {
      popover.style.left = Math.round((viewWidth - width) / 2) + "px";
      popover.style.top = Math.round(Math.max(EDGE, (viewHeight - height) / 2)) + "px";
      return true;
    }

    const rect = node.getBoundingClientRect();
    const box = {
      top: rect.top / zoom - PAD,
      left: rect.left / zoom - PAD,
      right: rect.right / zoom + PAD,
      bottom: rect.bottom / zoom + PAD,
    };
    hole.style.top = box.top + "px";
    hole.style.left = box.left + "px";
    hole.style.width = box.right - box.left + "px";
    hole.style.height = box.bottom - box.top + "px";

    const spaceBelow = viewHeight - box.bottom - GAP - EDGE;
    const spaceAbove = box.top - GAP - EDGE;
    const spaceRight = viewWidth - box.right - GAP - EDGE;
    const spaceLeft = box.left - GAP - EDGE;
    let top = box.bottom + GAP;
    let left = box.left;
    let fits = true;
    if (Math.max(spaceRight, spaceLeft) >= width) {
      // A full-width popover beside the target covers no page content: the tables
      // under the target stay readable while the walkthrough is open.
      top = box.top;
      left = spaceRight >= spaceLeft ? box.right + GAP : box.left - GAP - width;
    } else if (height <= spaceBelow) {
      top = box.bottom + GAP;
    } else if (height <= spaceAbove) {
      top = box.top - GAP - height;
    } else if (Math.max(spaceRight, spaceLeft) >= MIN_WIDTH) {
      popover.style.width = Math.min(width, Math.max(spaceRight, spaceLeft)) + "px";
      width = popover.offsetWidth;
      height = popover.offsetHeight;
      top = box.top;
      left = spaceRight >= spaceLeft ? box.right + GAP : box.left - GAP - width;
    } else {
      // No side has room. A shortened popover would hide "What to notice" behind a
      // scroll bar, so it keeps its full height and covers the bottom-right corner of
      // the target; the target's top (the tab heading) stays visible.
      fits = false;
      top = viewHeight - height - EDGE;
      left = box.right - width;
    }
    popover.style.left = clamp(left, EDGE, viewWidth - width - EDGE) + "px";
    popover.style.top = clamp(top, EDGE, viewHeight - height - EDGE) + "px";
    return fits && rect.top >= HEADER && rect.bottom <= window.innerHeight;
  }

  function schedule() {
    if (!frame) frame = window.requestAnimationFrame(place);
  }

  // Distance that brings the target's top to SCROLL_TOP. After a tab change Streamlit
  // scrolls the selected tab back into view, so the scroll stops where the selected tab
  // reaches the top edge: the page moves once instead of jumping back a second later.
  function scrollDistance(node) {
    const distance = node.getBoundingClientRect().top - SCROLL_TOP;
    const tab = selectedTabElement();
    return tab ? Math.min(distance, tab.getBoundingClientRect().top) : distance;
  }

  // Places the popover; when the target is not fully visible, scrolls it to the top of the
  // view once. The scroll event then places the popover again.
  function position() {
    resizeObserver.disconnect();
    const node = targetElement();
    if (node) resizeObserver.observe(node);
    if (!place() && node) {
      scrollContainer(node).scrollBy(0, scrollDistance(node));
    }
  }

  function waitForTarget(waited) {
    waitFrame = window.requestAnimationFrame(function () {
      waitFrame = 0;
      if (targetElement() || waited >= MAX_WAIT_FRAMES) position();
      else waitForTarget(waited + 1);
    });
  }

  function show(stepIndex) {
    index = stepIndex;
    const step = steps[index];
    count.textContent = step.tab
      ? "STEP " + pad(tabNumber(index)) + " / " + pad(tabStepCount) + " · " + step.keyword
      : step.keyword.toUpperCase();
    title.textContent = step.title;
    fill(what, step.what);
    fill(role, step.role);
    fill(code, step.code);
    fill(decision, step.decision);
    const backFocused = root.getRootNode().activeElement === back;
    back.disabled = index === 0;
    if (back.disabled && backFocused) next.focus({preventScroll: true});
    next.textContent = index === steps.length - 1 ? "Finish" : "Next";
    window.cancelAnimationFrame(waitFrame);
    waitFrame = 0;
    if (step.tab && selectTab(step.tab)) {
      hole.hidden = true;
      waitForTarget(0);
    } else {
      position();
    }
  }

  function move(offset) {
    const stepIndex = index + offset;
    if (stepIndex < 0) return;
    if (stepIndex === steps.length) dialog.close();
    else show(stepIndex);
  }

  function onKey(event) {
    if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
    event.preventDefault();
    move(event.key === "ArrowRight" ? 1 : -1);
  }

  function startListening() {
    if (listening) return;
    listening = true;
    window.addEventListener("scroll", schedule, true);
    window.addEventListener("resize", schedule);
    document.addEventListener("keydown", onKey);
  }

  function stopListening() {
    listening = false;
    window.removeEventListener("scroll", schedule, true);
    window.removeEventListener("resize", schedule);
    document.removeEventListener("keydown", onKey);
    window.cancelAnimationFrame(frame);
    window.cancelAnimationFrame(waitFrame);
    frame = 0;
    waitFrame = 0;
    resizeObserver.disconnect();
  }

  // SPOTLIGHT starts at the step of the open tab. On the first tab (the start of the
  // story) it starts with the overview.
  function startIndex(tabLabel) {
    const tabs = storyTabs();
    if (!tabLabel || (tabs.length && tabs[0].textContent.trim() === tabLabel)) return 0;
    const stepIndex = steps.findIndex(function (step) { return step.tab === tabLabel; });
    return stepIndex >= 0 ? stepIndex : 0;
  }

  function onStart() {
    const openTab = selectedTab();
    dialog.showModal();
    startListening();
    show(startIndex(openTab));
    next.focus({preventScroll: true});
  }

  // Esc, ×, Finish and the last Next all end in dialog.close(), which fires "close".
  // The page stays where the last card was: its tab remains open, nothing switches back.
  function onClose() {
    stopListening();
    start.focus({preventScroll: true});
  }

  function onBack() { move(-1); }
  function onNext() { move(1); }
  function onCloseClick() { dialog.close(); }

  start.addEventListener("click", onStart);
  back.addEventListener("click", onBack);
  next.addEventListener("click", onNext);
  close.addEventListener("click", onCloseClick);
  dialog.addEventListener("close", onClose);

  // Cleanup for component unmount and for a repeated render: no listener, frame or
  // observer survives; a walkthrough that is still open closes on the tab it shows.
  function cleanup() {
    dialog.removeEventListener("close", onClose);
    stopListening();
    if (dialog.open) dialog.close();
    root.removeEventListener(CLEANUP_EVENT, cleanup);
    root.remove();
  }
  root.addEventListener(CLEANUP_EVENT, cleanup);

  return cleanup;
}
