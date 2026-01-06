/* eslint-env browser */
// Listen for authorized_mcps event from backend and populate localStorage
(function () {
  console.log('[MCP Handler] Script loaded at', new Date().toISOString());

  // Function to populate MCPs
  function populateMCPs(mcps) {
    console.log('[MCP Handler] Populating MCPs:', mcps);

    const storageKey = 'mcp_storage_key';
    let existingMcps = [];

    try {
      const stored = localStorage.getItem(storageKey);
      console.log('[MCP Handler] Existing storage:', stored);
      if (stored) {
        existingMcps = JSON.parse(stored);
      }
    } catch (e) {
      console.error('[MCP Handler] Error parsing existing MCPs:', e);
    }

    // Create a set of existing MCP names to avoid duplicates
    const existingNames = new Set(
      existingMcps.map(function (mcp) {
        return mcp.name;
      })
    );
    console.log('[MCP Handler] Existing MCP names:', Array.from(existingNames));

    // Add new MCPs that don't already exist
    const newMcps = mcps.filter(function (mcp) {
      return !existingNames.has(mcp.name);
    });

    console.log('[MCP Handler] New MCPs to add:', newMcps);

    if (newMcps.length > 0) {
      const updatedMcps = existingMcps.concat(newMcps);
      localStorage.setItem(storageKey, JSON.stringify(updatedMcps));
      console.log('[MCP Handler] Updated MCPs in localStorage:', updatedMcps);

      // Force page reload to update React state
      console.log('[MCP Handler] Reloading page to update UI');
      setTimeout(function () {
        window.location.reload();
      }, 500);
    } else {
      console.log('[MCP Handler] No new MCPs to add');
    }
  }

  // Wait for socket.io to be available
  let attempts = 0;
  const maxAttempts = 100;

  const checkSocket = setInterval(function () {
    attempts++;

    if (window.socket && window.socket.connected) {
      console.log('[MCP Handler] Socket connected after', attempts, 'attempts');
      clearInterval(checkSocket);

      // Register listener for authorized_mcps event
      window.socket.on('authorized_mcps', function (data) {
        console.log('[MCP Handler] Received authorized_mcps event:', data);
        if (data.mcps && Array.isArray(data.mcps)) {
          populateMCPs(data.mcps);
        }
      });

      console.log(
        '[MCP Handler] Listener registered for authorized_mcps event'
      );
    } else if (attempts >= maxAttempts) {
      console.error(
        '[MCP Handler] Socket not found after',
        maxAttempts,
        'attempts'
      );
      clearInterval(checkSocket);
    }
  }, 100);
})();
