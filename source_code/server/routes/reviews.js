const router = require("express").Router();
let Review = require("../models/review.model");

//route to retrieve all the review (used only for development purpose)
router.route("/").get((req, res) => {
  Review.find()
    .then((reviews) => res.json(reviews))
    .catch((err) => res.status(400).json("error" + err));
});

//route to retrive all reviews of a restaurant given restaurant id
router.route("/:id").get((req, res) => {
  Review.find({ restaurantId: req.params.id })
    .then((reviews) => res.json(reviews))
    .catch((err) => res.status(400).json("errpr" + err));
});

//route to post a reviews of a restaurant given restaurant id and username
router.route("/:id").post((req, res) => {
  const restaurantId = req.params.id;
  const userid = req.body.userid;
  const username = req.body.username;
  const rating = Number(req.body.rating);
  const review = req.body.review;
  const newReview = new Review({
    restaurantId,
    userid,
    rating,
    review,
    username,
  });
  newReview
    .save()
    .then(() => res.json("Review added"))
    .catch((err) => res.status(400).json("error is " + err));
});

module.exports = router;
