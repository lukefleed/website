// KaTeX rendering script
// This script ensures KaTeX renders all math formulas reliably

(function() {
  'use strict';

  // Configuration for KaTeX auto-render
  const katexConfig = {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '$', right: '$', display: false },
      { left: '\\(', right: '\\)', display: false },
      { left: '\\[', right: '\\]', display: true }
    ],
    throwOnError: false,
    trust: true,
    strict: false,
    macros: {
      "\\RR": "\\mathbb{R}",
      "\\CC": "\\mathbb{C}",
      "\\NN": "\\mathbb{N}",
      "\\ZZ": "\\mathbb{Z}",
      "\\QQ": "\\mathbb{Q}"
    }
  };

  // Function to render all math on the page
  function renderAllMath() {
    if (typeof window.renderMathInElement === 'undefined') {
      console.warn('KaTeX auto-render not loaded yet, retrying...');
      return false;
    }

    try {
      window.renderMathInElement(document.body, katexConfig);
      console.log('KaTeX rendering completed successfully');
      return true;
    } catch (error) {
      console.error('KaTeX rendering error:', error);
      return false;
    }
  }

  // Function to wait for KaTeX to be loaded
  function waitForKatex(callback, maxAttempts = 50, interval = 100) {
    let attempts = 0;

    const checkKatex = setInterval(function() {
      attempts++;

      if (typeof window.renderMathInElement !== 'undefined') {
        clearInterval(checkKatex);
        callback();
      } else if (attempts >= maxAttempts) {
        clearInterval(checkKatex);
        console.error('KaTeX failed to load after', maxAttempts, 'attempts');
      }
    }, interval);
  }

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
      waitForKatex(renderAllMath);
    });
  } else {
    // DOM already loaded
    waitForKatex(renderAllMath);
  }

  // Export function for manual re-rendering if needed
  window.reRenderMath = function() {
    if (renderAllMath()) {
      console.log('Manual KaTeX re-render successful');
    } else {
      console.warn('Manual KaTeX re-render failed - KaTeX not loaded');
    }
  };

  // Handle dynamic content (for SPAs or dynamic page updates)
  if (typeof MutationObserver !== 'undefined') {
    let renderTimeout;
    const observer = new MutationObserver(function(mutations) {
      // Debounce rendering to avoid excessive calls
      clearTimeout(renderTimeout);
      renderTimeout = setTimeout(function() {
        const hasNewMath = mutations.some(function(mutation) {
          return Array.from(mutation.addedNodes).some(function(node) {
            if (node.nodeType === 1) { // Element node
              const text = node.textContent || '';
              return text.includes('$') || text.includes('\\(') || text.includes('\\[');
            }
            return false;
          });
        });

        if (hasNewMath && typeof window.renderMathInElement !== 'undefined') {
          renderAllMath();
        }
      }, 300);
    });

    // Start observing once KaTeX is loaded
    waitForKatex(function() {
      observer.observe(document.body, {
        childList: true,
        subtree: true
      });
    });
  }
})();
