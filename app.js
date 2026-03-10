const express = require("express");
const bodyParser = require("body-parser");
const path = require("path");
const fs = require("fs");
const app = express();
app.use(bodyParser.urlencoded({ extended: true }));
app.get("/", (req, res) => {
    res.sendFile(path.join(__dirname, "views", "index.html"));
});
app.post("/calculate", (req, res) => {
    const weight = parseFloat(req.body.weight);
    const height = parseFloat(req.body.height);
    if (!weight || !height || weight <= 0 || height <= 0) {
        return res.send("<h2>Enter valid positive values.</h2><a href='/'>Go Back</a>");
    }
    const bmi = weight / (height * height);
    let html = fs.readFileSync(
        path.join(__dirname, "views", "result.html"),
        "utf8"
    );
    html = html.replace(/{{BMI}}/g, bmi.toFixed(2));
    res.send(html);
});
app.listen(3000, () => {
    console.log("Server running at http://localhost:3000");
});