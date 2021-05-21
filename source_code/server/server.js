const express = require("express");
const cors = require("cors");
const mongoose = require("mongoose");

require("dotenv").config();

const app = express();
//setting up port number
const port = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

//setting up connection with mongodb atlas
const uri = process.env.ATLAS_URI;
mongoose.connect(uri, {
  useNewUrlParser: true,
  useCreateIndex: true,
  useUnifiedTopology: true,
});

const connection = mongoose.connection;

connection.once("open", () => {
  console.log("Mongodb connection established successfully");
});

const reviewsRouter = require("./routes/reviews");
const restaurantsRouter = require("./routes/restaurants");
app.use("/reviews", reviewsRouter);
app.use("/restaurants", restaurantsRouter);
app.listen(port, () => {
  console.log(`server is listening oon port: ${port}`);
});
