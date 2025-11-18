#!/bin/bash
# Configure proxy in webscraping config if PROXY_URL is set

CONFIG_FILE="/app/configs/nba/webscraping_config.yaml"

if [ -n "$PROXY_URL" ]; then
    echo "Configuring proxy: $PROXY_URL"

    # Add proxy-server to chrome_options if not already present
    if ! grep -q "proxy-server" "$CONFIG_FILE"; then
        # Find the line with --user-agent and add proxy after it
        sed -i "/--user-agent:/a\\  --proxy-server: \"$PROXY_URL\"" "$CONFIG_FILE"
        echo "Proxy configured successfully"
    else
        echo "Proxy already configured"
    fi
else
    echo "No PROXY_URL environment variable set, skipping proxy configuration"
fi
