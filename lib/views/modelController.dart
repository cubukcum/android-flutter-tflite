import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'TensorFlow Lite Example',
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  late tfl.Interpreter _interpreter;
  img.Image? _inputImage;
  img.Image? _outputImage;

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  Future<void> loadModel() async {
    _interpreter = await tfl.Interpreter.fromAsset('assets/your_model.tflite');
    setState(() {});
  }

  Future<void> performInference() async {
    if (_inputImage == null) return;

    // Resize image to required dimensions (256x256)
    img.Image resizedImage = img.copyResize(_inputImage!, width: 256, height: 256);

    // Convert image to byte buffer
    var input = resizedImage.getBytes();
    var output = List.filled(256 * 256 * 5, 0, growable: false).reshape([1, 256, 256, 5]);

    // Perform inference
    _interpreter.run(input, output);

    // Post-process output to apply color mapping
    img.Image outputImage = img.Image(256, 256);
    for (int y = 0; y < 256; y++) {
      for (int x = 0; x < 256; x++) {
        int maxIdx = 0;
        double maxValue = output[0][y][x][0];
        for (int c = 1; c < 5; c++) {
          if (output[0][y][x][c] > maxValue) {
            maxValue = output[0][y][x][c];
            maxIdx = c;
          }
        }
        outputImage.setPixel(x, y, getColorForClass(maxIdx) as int);
      }
    }

    setState(() {
      _outputImage = outputImage;
    });
  }

  Color getColorForClass(int classIdx) {
    switch (classIdx) {
      case 0:
        return Color.fromARGB(255, 64, 42, 42); // Road
      case 1:
        return Color.fromARGB(255, 255, 0, 0); // Lane
      case 2:
        return Color.fromARGB(255, 128, 128, 96); // Undrivable
      case 3:
        return Color.fromARGB(255, 0, 255, 102); // Movable
      case 4:
        return Color.fromARGB(255, 204, 0, 255); // Car
      default:
        return Colors.black;
    }
  }

  Future<void> loadImage() async {
    final ByteData data = await rootBundle.load('assets/input_image.jpg');
    final Uint8List bytes = Uint8List.view(data.buffer);
    _inputImage = img.decodeImage(bytes)!;
    performInference();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('TensorFlow Lite Example'),
      ),
      body: _outputImage != null
          ? Image.memory(Uint8List.fromList(img.encodePng(_outputImage!)))
          : Center(
        child: ElevatedButton(
          onPressed: loadImage,
          child: Text('Load Image and Perform Inference'),
        ),
      ),
    );
  }
}
