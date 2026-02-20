import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [brightness, setBrightness] = useState(0);
  const [width, setWidth] = useState("");
  const [height, setHeight] = useState("");
  const [filter, setFilter] = useState("none");
  const [imageUrl, setImageUrl] = useState(null);

  const handleSubmit = async () => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("filter_type", filter);
    formData.append("brightness", brightness);
    formData.append("width", width);
    formData.append("height", height);

    const response = await axios.post(
      "http://localhost:5000/api/image/process",
      formData,
      { responseType: "blob" }
    );

    setImageUrl(URL.createObjectURL(response.data));
  };

  return (
    <div style={{ padding: 30 }}>
      <h2>Image Editor 🚀</h2>

      <input type="file" onChange={(e) => setFile(e.target.files[0])} />
      <br /><br />

      <label>Brightness:</label>
      <input
        type="range"
        min="-100"
        max="100"
        value={brightness}
        onChange={(e) => setBrightness(e.target.value)}
      />

      <br /><br />

      <input
        placeholder="Width"
        onChange={(e) => setWidth(e.target.value)}
      />
      <input
        placeholder="Height"
        onChange={(e) => setHeight(e.target.value)}
      />

      <br /><br />

      <select onChange={(e) => setFilter(e.target.value)}>
        <option value="none">None</option>
        <option value="grayscale">Grayscale</option>
        <option value="blur">Blur</option>
        <option value="rotate">Rotate</option>
      </select>

      <br /><br />

      <button onClick={handleSubmit}>Process Image</button>

      <br /><br />

      {imageUrl && <img src={imageUrl} width="400" />}
    </div>
  );
}

export default App;