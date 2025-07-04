# Vibe Content Creation System

A modular, scalable content automation system for generating thousands of videos with synchronized audio, captions, and visual effects.

## 🎯 Project Overview

This system is designed to create engaging video content at scale, with features including:

- **Automated Video Generation**: Batch processing of video content with audio synchronization
- **Dynamic Caption System**: Multiple caption styles (cinematic, karaoke, minimal) with beat synchronization
- **Content Database**: Intelligent content selection and caching
- **AI-Powered Metadata**: Automatic tag and description generation using BLIP or OpenAI Vision
- **Modular Architecture**: Extensible pipeline system for different content types

## 🏗️ Project Structure

```
vibe-content-creation/
├── unified-video-system/        # Main video processing system
│   ├── audio/                   # Audio processing modules
│   ├── beat_sync/              # Beat detection and synchronization
│   ├── captions/               # Caption generation and styling
│   ├── content/                # Content database and selection
│   ├── core/                   # Core video processing engines
│   ├── output/                 # Generated video outputs
│   └── pipelines/              # Processing pipelines
├── MJAnime/                    # Anime-style video content
├── 11-scripts-for-tiktok/      # TikTok-specific audio scripts
└── docs/                       # Documentation and guides
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- FFmpeg
- Required Python packages (install with `pip3 install -r requirements.txt`)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vibe-content-creation.git
cd vibe-content-creation
```

2. Install dependencies:
```bash
pip3 install -r requirements.txt
```

3. Configure the system:
```bash
cp config/system_config.yaml.example config/system_config.yaml
# Edit the configuration file with your settings
```

### Basic Usage

1. **Single Video Generation**:
```bash
python3 unified-video-system/batch_generate_videos.py
```

2. **Batch Processing**:
```bash
python3 unified-video-system/batch_process_test.py
```

3. **Update Clip Metadata**:
```bash
python3 utils/update_clip_metadata.py
```

## 🎨 Features

### Video Processing
- **Multi-format Support**: MP4, MOV, AVI output formats
- **GPU Acceleration**: CUDA support for faster processing
- **Quality Optimization**: Automatic quality adjustment based on content

### Caption System
- **Multiple Styles**: Cinematic, karaoke, minimal caption presets
- **Beat Synchronization**: Captions sync with audio beats
- **Custom Fonts**: Support for custom font libraries
- **Real-time Preview**: Live caption preview during editing

### Content Management
- **Smart Caching**: Intelligent caching of processed content
- **Content Database**: Centralized content management system
- **Metadata Generation**: AI-powered tags and descriptions
- **Batch Operations**: Process multiple videos simultaneously

### Audio Processing
- **Beat Detection**: Advanced beat detection using librosa
- **Audio Sync**: Precise audio-visual synchronization
- **Multiple Formats**: WAV, MP3, M4A support
- **Volume Normalization**: Automatic audio level adjustment

## 🔧 Configuration

The system uses YAML configuration files in the `config/` directory:

- `system_config.yaml`: Main system configuration
- `caption_presets.json`: Caption styling presets
- `content_database_cache.json`: Content database cache

## 📊 Performance

- **Batch Processing**: Generate 10-100+ videos in parallel
- **GPU Acceleration**: 3-5x faster processing with CUDA
- **Smart Caching**: Reduced processing time for repeated content
- **Memory Optimization**: Efficient memory usage for large batches

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For questions, issues, or feature requests:

1. Check the [documentation](docs/)
2. Search existing [issues](https://github.com/yourusername/vibe-content-creation/issues)
3. Create a new issue if needed

## 🎯 Roadmap

- [ ] Cloud storage integration
- [ ] Real-time collaboration features
- [ ] Advanced AI content generation
- [ ] Mobile app companion
- [ ] API for third-party integrations

## 📈 Stats

- **Video Formats**: 3+ supported formats
- **Caption Styles**: 10+ preset styles
- **Processing Speed**: Up to 100 videos/hour
- **Content Types**: Unlimited with modular system

---

Built with ❤️ for content creators who want to scale their video production. 