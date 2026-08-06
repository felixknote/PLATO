# Example: NIS-Elements filenames + scrambled grid plate map

Real-world configuration for:

* Images named
  `WellA01_PointA01_0000_ChannelCam-DIA DIC Master Screening_Seq0000.tiff`
  (Nikon NIS-Elements, 2720x2720 uint16, single-channel DIC).
* A headerless 8x12 CSV plate map whose cells are `<condition>_<replicate>`,
  with randomised well positions and controls named `ACE-1 NC_n` / `MG1655 NC_n`.

Copy `plato.toml` next to your image folder and plate map, adjust
`[images].dir` if needed, then:

```bash
plato index && plato thumbs && plato gui
```

The plate map shipped here is the one this config was validated against.
