/*
SPDX-FileCopyrightText: 2026 GFZ Helmholtz Centre for Geosciences
SPDX-FileContributor: Sahil Jhawar

SPDX-License-Identifier: Apache-2.0
*/

/* Renders `.mermaid-pz` fences (see mkdocs.yml's custom_fences) and adds
 * mouse-wheel zoom + drag-to-pan (svg-pan-zoom, loaded as a global script
 * before this module). These are deliberately NOT plain `.mermaid` fences:
 * mkdocs-material's own built-in Mermaid support renders into a *closed*
 * shadow root (see its bundle.min.js), which is unreachable from any
 * outside script -- svg-pan-zoom included. Rendering it ourselves, straight
 * into the element's light DOM, is what makes the diagram scriptable. */
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";

mermaid.initialize({ startOnLoad: false });

let nextId = 0;

async function renderOne(el) {
  if (el.dataset.pzRendered) return;
  el.dataset.pzRendered = "true";

  const source = el.textContent;
  const { svg } = await mermaid.render(`mermaid-pz-${nextId++}`, source);
  el.innerHTML = svg;

  const svgEl = el.querySelector("svg");
  if (!svgEl) return;
  svgEl.style.maxWidth = "100%";

  const instance = svgPanZoom(svgEl, {
    zoomEnabled: true,
    panEnabled: true,
    controlIconsEnabled: true,
    fit: true,
    center: true,
    minZoom: 0.5,
    maxZoom: 15,
  });

  const resize = () => {
    instance.resize();
    instance.fit();
    instance.center();
  };
  window.addEventListener("resize", resize);
}

function scan() {
  document.querySelectorAll(".mermaid-pz").forEach((el) => {
    renderOne(el).catch((err) => console.error("mermaid-pz render failed:", err));
  });
}

if (window.document$) {
  // Material for MkDocs' navigation observable: fires on every page,
  // including instant-loading client-side transitions.
  document$.subscribe(scan);
} else {
  document.addEventListener("DOMContentLoaded", scan);
}
