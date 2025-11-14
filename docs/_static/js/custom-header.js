// Compute the path to the root, then append _static/wizarddawg.png
function getLogoPath() {
    var path = window.location.pathname;
    var depth = (path.match(/\//g) || []).length - 2; // -2: remove leading slash and filename
    var prefix = "";
    if (depth > 0) {
        prefix = "../".repeat(depth);
    }
    return prefix + "_static/wizarddawg.png";
}

document.addEventListener("DOMContentLoaded", function () {
    // Previously this code created an <a> linking to the old docs site.
    // That behaviour has been removed so the logo is no longer a link.
    const logo = document.createElement("img");
    logo.id = "floating-logo";
    //logo.src = getLogoPath();
    logo.src = "/_static/wizarddawg.png";
    logo.alt = "Logo";

    // Append the image directly to the body (not wrapped in a link).
    document.body.appendChild(logo);
});