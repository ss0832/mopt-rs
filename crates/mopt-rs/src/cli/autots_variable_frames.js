// Play one saved path in its native variables. A projection changes only the drawing.
for (const panel of document.querySelectorAll(".variable-path")) {
    const controls = panel.querySelector(".controls");
    const link = controls.querySelector("a[download]");
    const bytes = Uint8Array.from(atob(link.href.split(",")[1]), c => c.charCodeAt(0));
    const lines = new TextDecoder().decode(bytes).trim().split(/\r?\n/);
    const frames = [];
    const names = [];
    for (let i = 0; i < lines.length;) {
        const count = Number(lines[i++]);
        const comment = lines[i++];
        const values = [];
        for (let j = 0; j < count; j++) {
            const fields = lines[i++].trim().split(/\s+/);
            values.push(Number(fields.pop()));
            if (!frames.length) names.push(fields.join(" "));
        }
        frames.push({values, comment});
    }
    if (!names.length) continue;

    const add = (tag, parent, label) => {
        const element = document.createElement(tag);
        if (label !== undefined) element.textContent = label;
        parent.appendChild(element);
        return element;
    };
    const axis = (label, initial) => {
        const select = add("select", add("label", controls, label + " "));
        ["Point", ...names].forEach(name => add("option", select, name));
        select.selectedIndex = initial;
        select.onchange = draw;
        return select;
    };
    const xAxis = axis("X", names.length > 1 ? 1 : 0);
    const yAxis = axis("Y", names.length > 1 ? 2 : 1);
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 640 240");
    svg.setAttribute("role", "img");
    panel.insertBefore(svg, controls);
    function tick(x, y, anchor) {
        const text = document.createElementNS(ns, "text");
        text.setAttribute("x", x);
        text.setAttribute("y", y);
        text.setAttribute("text-anchor", anchor);
        text.setAttribute("fill", "var(--muted)");
        text.setAttribute("font-size", "12");
        svg.appendChild(text);
        return text;
    }
    const xLow = tick(52, 232, "start"), xHigh = tick(624, 232, "end");
    const yLow = tick(46, 208, "end"), yHigh = tick(46, 20, "end");
    const line = document.createElementNS(ns, "polyline");
    line.setAttribute("fill", "none");
    line.setAttribute("stroke", "var(--accent)");
    line.setAttribute("stroke-width", "2");
    svg.appendChild(line);
    const marker = document.createElementNS(ns, "circle");
    marker.setAttribute("r", "6");
    marker.setAttribute("fill", "var(--s-red)");
    svg.appendChild(marker);

    const play = add("button", controls, "Play");
    const slider = add("input", controls);
    slider.type = "range";
    slider.min = 0;
    slider.max = frames.length - 1;
    slider.value = 0;
    slider.setAttribute("aria-label", "Path frame");
    const caption = add("output", controls);
    const pointLink = add("a", controls, "Download frame (.txt)");
    pointLink.className = "dl";
    const readout = add("div", panel);
    readout.className = "table-wrap";
    const table = add("table", readout);
    const heads = add("tr", add("thead", table));
    names.forEach(name => add("th", heads, name));
    const row = add("tr", add("tbody", table));
    const cells = names.map(() => add("td", row));
    let projected = [];
    function draw() {
        const scaled = (axis, from, to, low, high) => {
            const index = axis.selectedIndex - 1;
            const values = frames.map((f, i) => index < 0 ? i + 1 : f.values[index]);
            let lo = Infinity, hi = -Infinity;
            for (const v of values) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
            low.textContent = lo.toPrecision(3);
            high.textContent = hi.toPrecision(3);
            return values.map(v => from + (hi === lo ? 0.5 : (v - lo) / (hi - lo)) * (to - from));
        };
        const x = scaled(xAxis, 52, 624, xLow, xHigh), y = scaled(yAxis, 208, 16, yLow, yHigh);
        projected = x.map((v, i) => [v, y[i]]);
        line.setAttribute("points", projected.map(p => p.join(",")).join(" "));
        svg.setAttribute("aria-label", xAxis.selectedOptions[0].text + " against " + yAxis.selectedOptions[0].text);
        show();
    }
    function show() {
        const i = Number(slider.value), frame = frames[i];
        marker.setAttribute("cx", projected[i][0]);
        marker.setAttribute("cy", projected[i][1]);
        caption.textContent = "Frame " + (i + 1) + " / " + frames.length + " · " + frame.comment.split("energy=").pop();
        frame.values.forEach((v, j) => { cells[j].textContent = v.toPrecision(8); });
        const point = names.map((n, j) => n + " " + frame.values[j]).join("\n");
        pointLink.href = "data:text/plain;charset=utf-8," + encodeURIComponent(point + "\n");
        pointLink.download = link.download.replace(/\.txt$/, "_frame_" + (i + 1) + ".txt");
    }
    slider.oninput = show;
    let timer = null;
    play.disabled = frames.length === 1;
    play.onclick = () => {
        if (timer !== null) { clearInterval(timer); timer = null; play.textContent = "Play"; }
        else {
            play.textContent = "Pause";
            timer = setInterval(() => { slider.value = (Number(slider.value) + 1) % frames.length; show(); }, 180);
        }
    };
    draw();
}
