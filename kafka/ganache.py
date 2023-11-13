from web3 import Web3

# Define the URL of the Ganache CLI JSON-RPC endpoint
ganache_url = "http://ganache-cli-mktetts-dev.apps.sandbox-m3.1530.p1.openshiftapps.com"  # Update with the actual ClusterIP or service URL

# Create a Web3 instance
w3 = Web3(Web3.HTTPProvider(ganache_url))

# Check if connected to Ganache CLI
try:
    # Check if connected to Ganache CLI
    w3.provider.is_connected()
    print("Connected to Ganache CLI")

    # Get the list of available accounts
    accounts = w3.eth.accounts
    print(accounts)
    if len(accounts) > 0:
        print("Available accounts:")
        for account in accounts:
            balance = w3.from_wei(w3.eth.get_balance(account), 'ether')
            print(f"Account: {account}, Balance: {balance} ETH")
    else:
        print("No accounts found.")

except ConnectionError:
    print("Failed to connect to Ganache CLI")

