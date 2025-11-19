#!/bin/bash
# Configure proxy in webscraping config if PROXY_URL is set

set -e

CONFIG_FILE="/app/configs/nba/webscraping_config.yaml"

if [ -n "$PROXY_URL" ]; then
    echo "=== Proxy Configuration ==="
    echo "PROXY_URL is set to: $PROXY_URL"

    # Validate proxy URL format
    if [[ ! "$PROXY_URL" =~ ^https?:// ]]; then
        echo "WARNING: PROXY_URL doesn't start with http:// or https://"
        echo "This may cause Chrome to fail. Skipping proxy configuration."
        exit 0
    fi

    # Check if config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "ERROR: Config file not found at $CONFIG_FILE"
        exit 1
    fi

    # Add proxy-server to chrome_options if not already present
    if ! grep -q "proxy-server" "$CONFIG_FILE"; then
        echo "Adding --proxy-server to Chrome options..."
        # Find the line with --user-agent and add proxy after it
        sed -i "/--user-agent:/a\\  --proxy-server: \"$PROXY_URL\"" "$CONFIG_FILE"
        echo "✓ Proxy configured successfully"

        # Show the added line for verification
        echo "Verifying proxy configuration:"
        grep -A1 -B1 "proxy-server" "$CONFIG_FILE" || echo "WARNING: Could not verify proxy in config"
    else
        echo "Proxy already configured in config file"
    fi

    # Test if proxy is reachable (optional, with timeout)
    echo "Testing proxy connectivity..."
    if timeout 5 curl -x "$PROXY_URL" -s -o /dev/null -w "%{http_code}" https://www.google.com 2>&1 | grep -q "200\|301\|302"; then
        echo "✓ Proxy is reachable"
    else
        echo "WARNING: Proxy test failed or timed out. Chrome may fail to start."
        echo "Consider disabling proxy if webscraping fails."
    fi
else
    echo "No PROXY_URL environment variable set, skipping proxy configuration"
fi

echo "=== Proxy Configuration Complete ==="
