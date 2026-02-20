const axios = require("axios");
const FormData = require("form-data");

exports.processImage = async (req, res) => {
  try {
    const form = new FormData();
    form.append("file", req.file.buffer, req.file.originalname);
    form.append("filter_type", req.body.filter_type);
    form.append("width", req.body.width);
    form.append("height", req.body.height);
    form.append("brightness", req.body.brightness);

    const response = await axios.post(
      "http://localhost:8000/process",
      form,
      {
        headers: form.getHeaders(),
        responseType: "arraybuffer"
      }
    );

    res.set("Content-Type", "image/jpeg");
    res.send(response.data);

  } catch (error) {
    console.error(error);
    res.status(500).send("Error processing image");
  }
};