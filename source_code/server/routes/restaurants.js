const router = require("express").Router();
const axios = require("axios");
const { json } = require("express");
const fs = require("fs");
const { getMaxListeners } = require("process");
//importing json file created by python
let rawdata = fs.readFileSync("reccomended_restaurants.json");
let recommended_restaurants = JSON.parse(rawdata);
const array = recommended_restaurants["Nishant Pathare"];
//route for main landing page to return the restaurants given
router.route("/cate/:zipcode/:category").get((req, res) => {
  axios
    .get(
      `https://api.yelp.com/v3/businesses/search?location=${req.params.zipcode}`,
      {
        headers: {
          Authorization: `Bearer UwwtBJxyBTvumTkjXkvDlcBOKg9U9sjo87e6Pizxqn5t957RgOVnoKYrIc1CaFbj3sMJSeS_kyFY3k9BoF5Hiz0fY6sw5VRkTMU6dg9JDfGoHAG6ZVvR0ALJzBp3YHYx`,
        },
        params: {
          categories: `${req.params.category}`,
          limit: 6,
          sort_by: "review_count",
        },
      }
    )
    .then((ress) => res.json(ress.data))
    .catch((err) => res.status(400).json("errpr" + err));
});

//route for main landing page to return highest rated restauarnts
router.route("/highrating/:zipcode/:category").get((req, res) => {
  axios
    .get(
      `https://api.yelp.com/v3/businesses/search?location=${req.params.zipcode}`,
      {
        headers: {
          Authorization: `Bearer UwwtBJxyBTvumTkjXkvDlcBOKg9U9sjo87e6Pizxqn5t957RgOVnoKYrIc1CaFbj3sMJSeS_kyFY3k9BoF5Hiz0fY6sw5VRkTMU6dg9JDfGoHAG6ZVvR0ALJzBp3YHYx`,
        },
        params: {
          categories: `${req.params.category}`,
          limit: 6,
          sort_by: "rating",
        },
      }
    )
    .then((ress) => res.json(ress.data))
    .catch((err) => res.status(400).json("errpr" + err));
});

//route for getting particular restaurant data given restaurant id
router.route("/:restid").get((req, res) => {
  axios
    .get(`https://api.yelp.com/v3/businesses/${req.params.restid}`, {
      headers: {
        Authorization: `Bearer UwwtBJxyBTvumTkjXkvDlcBOKg9U9sjo87e6Pizxqn5t957RgOVnoKYrIc1CaFbj3sMJSeS_kyFY3k9BoF5Hiz0fY6sw5VRkTMU6dg9JDfGoHAG6ZVvR0ALJzBp3YHYx`,
        "Content-Type": "application/json",
      },
    })
    .then((ress) => {
      res.json(ress.data);
    })
    .catch((err) => res.status(400).json("errpr" + err));
});

router.route("/recommendations/:id").get((req, res) => {
  let promises = [];
  array.forEach((key) => {
    console.log("key is", key);
    promises.push(
      axios.get(`https://api.yelp.com/v3/businesses/${key}`, {
        headers: {
          Authorization: `Bearer UwwtBJxyBTvumTkjXkvDlcBOKg9U9sjo87e6Pizxqn5t957RgOVnoKYrIc1CaFbj3sMJSeS_kyFY3k9BoF5Hiz0fY6sw5VRkTMU6dg9JDfGoHAG6ZVvR0ALJzBp3YHYx`,
          "Content-Type": "application/json",
        },
      })
    );
  });
  Promise.all(promises).then((responses) => {
    console.log(responses);
  });
  res.json({ hi: "dds" });
});

//route for sending back restaurant data required for plotting charts
router.route("/charts/:zipcode").get((req, res) => {
  //1,1.5,2,2.5,3,3.5,4,4.5,5
  var ratings = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  //$,$$,$$$
  var pricing = [0, 0, 0];
  //breakfast_brunch,burgers,cafes,chinese,hotdog,pizza,seafood,steak,coffee
  var categories = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  axios
    .get(
      `https://api.yelp.com/v3/businesses/search?location=${req.params.zipcode}`,
      {
        headers: {
          Authorization: `Bearer UwwtBJxyBTvumTkjXkvDlcBOKg9U9sjo87e6Pizxqn5t957RgOVnoKYrIc1CaFbj3sMJSeS_kyFY3k9BoF5Hiz0fY6sw5VRkTMU6dg9JDfGoHAG6ZVvR0ALJzBp3YHYx`,
        },
        params: { limit: 50 },
      }
    )

    .then((ress) => {
      ress.data.businesses.forEach((item) => {
        if (item.rating === 1) ratings[8]++;
        else if (item.rating === 1.5) ratings[7]++;
        else if (item.rating === 2) ratings[6]++;
        else if (item.rating === 2.5) ratings[5]++;
        else if (item.rating === 3) ratings[4]++;
        else if (item.rating === 3.5) ratings[3]++;
        else if (item.rating === 4) ratings[2]++;
        else if (item.rating === 4.5) ratings[1]++;
        else if (item.rating === 5) ratings[0]++;

        if (item.price === "$") pricing[2]++;
        else if (item.price === "$$") pricing[1]++;
        else if (item.price === "$$$") pricing[0]++;
        item.categories.map(({ alias }) => {
          if (alias === "breakfast_brunch") categories[0]++;
          else if (alias === "burgers") categories[1]++;
          else if (alias === "cafes") categories[2]++;
          else if (alias === "chinese") categories[3]++;
          else if (alias === "hotdog") categories[4]++;
          else if (alias === "pizza") categories[5]++;
          else if (alias === "seafood") categories[6]++;
          else if (alias === "steak") categories[7]++;
          else if (alias === "coffee") categories[8]++;
        });
      });

      //function below returns top n values from an object array
      const topN = (arr, n) => {
        if (n > arr.length) {
          return false;
        }
        return arr
          .slice()
          .sort((a, b) => {
            return b.review_count - a.review_count;
          })
          .slice(0, n);
      };
      const topreviews = topN(ress.data.businesses, 5);

      const returnobject = {
        topreviews,
        ratings,
        pricing,
        categories,
        data: ress.data.businesses,
      };
      res.json(returnobject);
    })
    .catch((err) => res.status(400).json("errpr" + err));
});

//jhol
router.route("/jhol/:zipcode/:category").get((req, res) => {
  axios
    .get(
      `https://api.yelp.com/v3/businesses/search?location=${req.params.zipcode}`,
      {
        headers: {
          Authorization: `Bearer UwwtBJxyBTvumTkjXkvDlcBOKg9U9sjo87e6Pizxqn5t957RgOVnoKYrIc1CaFbj3sMJSeS_kyFY3k9BoF5Hiz0fY6sw5VRkTMU6dg9JDfGoHAG6ZVvR0ALJzBp3YHYx`,
        },
        params: {
          categories: `${req.params.category}`,
          limit: 4,
          attributes: "hot_and_new",
        },
      }
    )
    .then((ress) => res.json(ress.data))
    .catch((err) => res.status(400).json("errpr" + err));
});

//This route is for sending back nearest restaurants
router.route("/nearby/:lat/:lng/:category").get((req, res) => {
  axios
    .get(
      `https://api.yelp.com/v3/businesses/search?latitude=${req.params.lat}&longitude=${req.params.lng}`,
      {
        headers: {
          Authorization: `Bearer UwwtBJxyBTvumTkjXkvDlcBOKg9U9sjo87e6Pizxqn5t957RgOVnoKYrIc1CaFbj3sMJSeS_kyFY3k9BoF5Hiz0fY6sw5VRkTMU6dg9JDfGoHAG6ZVvR0ALJzBp3YHYx`,
        },
        params: {
          categories: `${req.params.category}`,
          limit: 6,
          sort_by: "distance",
        },
      }
    )
    .then((ress) => res.json(ress.data))
    .catch((err) => res.status(400).json("errpr" + err));
});

module.exports = router;
