import React, { useState, useRef, useEffect } from "react";
import "./App.css";
import grey from './assets/grey.png';

function App() {
  const [imageInput, setImageInput] = useState(null);
  const [imageOutput, setImageOutput] = useState(null);
  const [loaderOn, setLoaderOn] = useState(false);
  const [drawing, setDrawing] = useState(false);
  const [brushSize, setBrushSize] = useState(5);
  const [maskedPositions, setMaskedPositions] = useState([]);
  const canvasRef = useRef(null);

  useEffect(() => {
    if (imageInput) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      const img = new Image();
      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      };
      img.src = URL.createObjectURL(imageInput);
    }
  }, [imageInput]);

  const handleImageChange = (event) => {
    setImageInput(event.target.files[0]);
    setImageOutput(grey);
  };

  const handleClearOutput = (event) => {
    setImageOutput(grey);
  };

  const handleMouseDown = () => {
    setDrawing(true);
  };

  const handleMouseUp = () => {
    setDrawing(false);
  };

  const handleMouseMove = (event) => {
    if (!drawing) {
      return;
    }
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const x = event.nativeEvent.offsetX;
    const y = event.nativeEvent.offsetY;
    ctx.fillStyle = "red";
    ctx.beginPath();
    ctx.arc(x, y, brushSize, 0, 2 * Math.PI);
    ctx.fill();
    const newMaskedPositions = [];
    for (let dy = -brushSize; dy <= brushSize; dy++) {
      for (let dx = -brushSize; dx <= brushSize; dx++) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx >= 0 && ny >= 0 && nx < canvas.width && ny < canvas.height && !newMaskedPositions.some((pos) => pos.x === nx && pos.y === ny)) {
          newMaskedPositions.push({ x: nx, y: ny });
        }
      }
    }
    setMaskedPositions((prevState) => [...prevState, ...newMaskedPositions]);
  };

  const genWrapper = () => {
    setLoaderOn(true);
    setTimeout(handleGenerate, 1000);
  }

  const handleGenerate = async () => {
    const canvas = canvasRef.current;
    var h = 0;
    var w = 0;
    if (!canvas) {
      h = 256;
      w = 256;
    } else {
      h = canvas.height;
      w = canvas.width;
    }
    var mask = Array.from(Array(h), () => Array(w).fill(0));
    for (let p = 0; p < maskedPositions.length; p++) {
      mask[maskedPositions[p].y][maskedPositions[p].x] = 1;
    }
    const formData = new FormData();
    formData.append("image", imageInput);
    formData.append("mask", JSON.stringify(mask));
    try {
      const response = await fetch("http://localhost:9000/upload.php", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      console.log(data.output);
      setImageOutput(`data:image/jpeg;base64,${data.image}`);
    } catch (error) {
      console.error(error);
    } finally {
      setLoaderOn(false);
    }
  };

  const handleClearMask = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (imageInput) {
      const img = new Image();
      img.onload = () => {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      };
      img.src = URL.createObjectURL(imageInput);
    }
    setMaskedPositions([]);
  };

  const handleBrushSizeChange = (event) => {
    setBrushSize(parseInt(event.target.value, 10));
  };

  return (
    <div className="App">
      <h1>EDIFFUSER</h1>
      <div>
        <label htmlFor="file-upload" className="filePicker">
          <i className="fa fa-cloud-upload"></i> Select Image
        </label>
        <input id="file-upload" type="file" onChange={handleImageChange} />
      </div>
      <div className="images">
        {loaderOn && <div className="overlay">
          <div className="loader"></div>
        </div>}
        {imageInput && (
            <canvas className="imin" ref={canvasRef} onMouseDown={handleMouseDown} onMouseUp={handleMouseUp} onMouseMove={handleMouseMove} />
        )}
        {imageOutput && <img className="imout" src={imageOutput} alt="Output" />}
      </div>
      <div className="brush-slider">
        <label className="brushlabel" htmlFor="brush-size">Brush size: </label>
        <input className="slider" type="range" id="brush-size" name="brush-size" min="1" max="50" value={brushSize} onChange={handleBrushSizeChange} />
      </div>
      <div>
        <button className="genButton" onClick={genWrapper}>
          Generate
        </button>
        <button className="clearButton" onClick={handleClearMask}>
          Clear Mask
        </button>
        <button className="clearOutButton" onClick={handleClearOutput}>
          Clear Output
        </button>
      </div>
    </div>
  );
}

export default App;
