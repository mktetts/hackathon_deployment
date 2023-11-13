from confluent_kafka import Producer

# Kafka broker(s) list
bootstrap_servers = "10.128.17.50:9092"

# Create a Kafka producer configuration
producer_config = {
    'bootstrap.servers': bootstrap_servers,
}

# Create a producer instance
producer = Producer(producer_config)

# Define a topic to produce messages to
topic = "your-topic-name"

# Define the message you want to send
message_key = "key"
message_value = "Hello, Kafka!"

# Produce the message to the topic
producer.produce(topic, key=message_key, value=message_value)

# Wait for any outstanding messages to be delivered and delivery reports to be received
producer.flush()
