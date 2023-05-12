<?php
// Set the Access-Control-Allow-Origin header to allow CORS
header('Access-Control-Allow-Origin: *');

// Set the allowed headers for CORS
header('Access-Control-Allow-Headers: Content-Type');

// Check if this is a preflight request
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
  // Set the allowed methods for CORS
  header('Access-Control-Allow-Methods: POST, OPTIONS');
  exit();
}

// Retrieve image file and binary mask from POST request
$imageFile = $_FILES['image'];
$mask = json_decode($_POST['mask'], true);

// Save the image file to a directory on the server
$uploadsDir = './inputs/';
$imagePath = $uploadsDir . basename($imageFile['name']);

// Save the binary mask to a file on the server
$maskDir = './inputs/';
$maskPath = $maskDir . basename($imageFile['name'], '.' . pathinfo($imageFile['name'], PATHINFO_EXTENSION)) . '.json';
file_put_contents($maskPath, json_encode($mask));

// Call Python script with image and mask as arguments
$pythonScript = './test.py';
$imageArg = escapeshellarg($imagePath);
$maskArg = escapeshellarg($maskPath);
$command = "python $pythonScript $imageArg $maskArg";
$output = shell_exec($command);
error_log("Python script output: ".$output);

$output_path = "./outputs/xsamp.jpg";
$response = array(
  'message' => 'Data received successfully',
  'image' => base64_encode(file_get_contents($output_path)),
  'output' => $output
);
echo json_encode($response);
?>
