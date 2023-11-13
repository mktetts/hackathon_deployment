const express = require("express");
const { ExpressPeerServer } = require("peer");
const cors = require("cors");

const app = express();



app.use(cors());



const server = app.listen(3000, () => {
    console.log("Peer is running on 3000");
});

const peerServer = ExpressPeerServer(server, {
    debug: true,
});

app.use("/peerjs", peerServer);

let Client = null;
peerServer.on("connection", (client) => {
    client.send({
        user: "Connected with id : " + client.id,
    });
    Client = client;
    client.socket.on("message", (data) => {
        var dataString = data.toString("utf8");
        dataString = JSON.parse(dataString);
        if (dataString.type === "HEARTBEAT") {
        } else {
            // Process and respond to other types of messages
            if (dataString.for === "profileView") {
                profileView(dataString);
            } else if (dataString.for === "userCreate") {
                userCreate(dataString);
            }
        }
    });
});
// peerServer.on("disconnect", (client) => {
//     console.log("client disconnected");
// });

