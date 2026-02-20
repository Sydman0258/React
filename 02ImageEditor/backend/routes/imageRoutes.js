const express = require("express");
const router = express.Router();
const upload = require("multer")();
const { processImage } = require("../controller/imageController");

router.post("/process", upload.single("file"), processImage);

module.exports = router;