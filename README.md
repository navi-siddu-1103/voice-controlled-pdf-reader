# voice-controlled-pdf-reader

A Python application that allows users to navigate PDF documents using voice commands. The application uses machine learning to recognize commands like "scroll up", "scroll down", "move up", "move down", etc.

## Features

- **Voice Command Recognition**: Uses speech recognition with ML to understand scroll commands
- **Continuous Scrolling**: Auto-scrolls in the specified direction until told to stop
- **Adjustable Speed**: Control how fast the document scrolls
- **PDF Navigation**: Jump to top/bottom of document with voice commands
- **Attractive UI**: Modern, clean interface with visual feedback
- **Command Logging**: Shows your recent voice commands

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/voice-pdf-scroller.git
   cd voice-pdf-scroller
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

   Or install the package:
   ```bash
   pip install .
   ```

## Usage

1. Start the application:
   ```bash
   python main.py
   ```
   
   Or if installed as a package:
   ```bash
   voice-pdf
   ```

2. Click "Open PDF" to select your document
3. Click "Start Voice Control" to enable voice commands
4. Speak commands to navigate the document

## Voice Commands

- **"Up"**, **"Scroll Up"**, **"Move Up"**: Scroll upward continuously
- **"Down"**, **"Scroll Down"**, **"Move Down"**: Scroll downward continuously
- **"Stop"**: Stop scrolling
- **"Top"**: Jump to document beginning
- **"Bottom"**: Jump to document end
- **"Exit"**: Close the application

## Development

### Project Structure
```
voice_pdf_scroller/
│
├── main.py                 # Main entry point
├── requirements.txt        # Dependencies list
├── setup.py                # Package installation script
├── resources/
│   └── icon.png            # Application icon
│
├── src/
│   ├── __init__.py
│   ├── ui/                 # User interface components
│   ├── pdf/                # PDF document handling
│   ├── speech/             # Speech recognition & ML
│   └── utils/              # Utility functions
│
└── tests/                  # Unit tests
```

### Running Tests

```bash
python -m unittest discover -s tests
```

## Requirements

- Python 3.6 or higher
- PyQt5 for the user interface
- SpeechRecognition for voice input
- scikit-learn for machine learning
- numpy for data processing

## License

This project is licensed under the MIT License - see the LICENSE file for details.
