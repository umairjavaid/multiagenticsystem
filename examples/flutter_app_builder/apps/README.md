# Flutter Apps Directory

This directory contains Flutter applications created by the Flutter App Builder system.

## Structure

Each Flutter app created by the builder will have its own subdirectory:

```
apps/
├── app_name_1/
│   ├── lib/
│   ├── pubspec.yaml
│   └── ...
├── app_name_2/
│   ├── lib/
│   ├── pubspec.yaml
│   └── ...
└── README.md (this file)
```

## Usage

Flutter apps are created here when you run:
- `flutter_builder.py` - Direct Python script
- `config_driven_builder.py` - Configuration-driven approach

The system ensures all Flutter projects are created in this directory, maintaining proper organization and preventing clutter in the root workspace.
