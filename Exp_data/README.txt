GENERAL INFORMATION

1. Title of dataset: Digital image correlation measurement of linear elastic steel specimen

2. Author/ Creator information
    Principal Creator Contact Information: Jendrik-Alexander Tröger, Institute of Applied Mechanics, Clausthal University of Technology, Adolph-Roemer-Str. 2a, 38678 Clausthal-Zellerfeld, Germany, jendrik-alexander.troeger@tu-clausthal.de
    Co-creator Contact Information: Stefan Hartmann, Institute of Applied Mechanics, Clausthal University of Technology, Adolph-Roemer-Str. 2a, 38678 Clausthal-Zellerfeld, Germany, stefan.hartmann@tu-clausthal.de
    Co-creator Contact Information: David Anton, Institute for Computational Modeling in Civil Engineering, Technische Universität Braunschweig, Pockelsstr. 3, 38106 Braunschweig, Germany, d.anton@tu-braunschweig.de
    Co-creator Contact Information: Henning Wessels, Institute for Computational Modeling in Civil Engineering, Technische Universität Braunschweig, Pockelsstr. 3, 38106 Braunschweig, Germany, h.wessels@tu-braunschweig.de

3. Project: Deterministic and statistical calibration of constitutive models from full-field data with parametric physics-informed neural networks

4. Date of data collection: 2023-11-16

5. Geographic location of data collection: Solid Mechanics laboratory, Institute of Applied Mechanics, Clausthal University of Technology, Adolph-Roemer-Str. 2a, 38678 Clausthal-Zellerfeld, Germany

6. Information about funding sources that supported the collection of the data: No funding applicable


ABSTRACT

The dataset comprises the axial and lateral displacements on the surface of a plate with a hole subjected to tensile load. The displacement data are measured by digital image correlation and the material is assumed to behave linear elastic. The material under investigation is a common low-carbon steel alloy of type S235. The displacement data are used for calibration of a linear elastic constitutive model using parametric physics-informed neural networks and finite elements. For that purpose, the dataset comprises both the raw experimental displacement data and displacement data interpolated onto a regular grid using linear interpolation, where the interpolation routine is provided as well.


SHARING/ACCESS INFORMATION

1. Licenses or restrictions placed on the data: Creative Commons Attribution 4.0 International

2. Was data derived from another source?: No

3. Links to publications that cite or use the data: See metadata on zenodo.org


METHODOLOGICAL INFORMATION

1. Description of methods used for collection or generation of data:
The data were obtained from a displacement-driven tensile test on a plate with a hole, which was carried out on a Z100 testing machine from ZwickRoell GmbH & Co. KG. The prescribed displacement velocity of the traverse was 0.01 mm/s with a maximum traverse displacement of 0.6 mm. A speckle pattern consisting of a white primer and black dots was applied to the measurement region on the specimen's surface before performing the tensile experiment to enable measuring surface displacements using digital image correlation. For the present data, we employed the digital image correlation system ARAMIS 12M from Carl Zeiss GOM Metrology GmbH and evaluated the captured images using the software ARAMIS Professional 2020. The resolution of the cameras is stated by the manufacturer as 4096x3000 px. During the analysis, we used Titanar 2.8/50 lenses with focal length 50 mm. Moreover, a light projector using narrow-band blue light technology was used for the illumination and an aperture of 16 was chosen. The working distance between digital image correlation system and specimen was 564 mm and the camera angle 25°. The digital image correlation system was calibrated using a CP40/100 calibration object by following the calibration instructions in the software ARAMIS Professional 2020. During the tensile test, image capturing was carried out with 1 Hz.

2. Methods for processing the data: 
The displacement data were extracted from the last load-step, that is, a prescribed traverse displacement of 0.6 mm after 60 s. The raw images were correlated using a surface component in ARAMIS Professional 2020 with 25 px facet size and 20 px point distance employing default settings for the computation of the image matching versus the initial stage. Moreover, the rigid body movements due to the testing machine's limited stiffness were considered using two small scales, which were connected to the clampings. The compensation of rigid body movements was carried out using the corresponding option in ARAMIS Professional 2020. The displacements obtained from the ARAMIS Professional 2020 software are denoted as raw displacements in the dataset. This data are the basis for computing the interpolated displacement values on a regular grid, which is used during calibration. Only displacements in the parallel area of the specimen are taken into account for the interpolation. The dataset contains an image of the specimen geometry in which this area is marked by dashed lines. The interpolation was carried out with the Matlab-function scatteredInterpolant using default settings. The regular grid defining the target for the interpolation is composed of common 8-noded quadrilateral elements. The stress boundary condition for the calibration was obtained from the axial force measured at the testing machine under consideration of the specimen's width of 20.05 mm and thickness of 3.05 mm.

3. Instrument- or software-specific information needed to interpret the data:
The image matching was done using ARAMIS Professional 2020 from Carl Zeiss GOM Metrology GmbH. The interpolation was done using Matlab R2022b and the function scatteredInterpolant.

4. Standards and calibration information, if appropriate: 
The geometry of the tensile specimen is broadly similar to DIN EN ISO 6892-1, except the length between the clampings and the insertion of a hole centered in length and width. The dataset contains an image of the geometry.

5. Environmental/experimental conditions:
The tensile test was performed at approximately 20 °C within a common environment of a material's testing laboratory in engineering sciences.

6. Describe any quality-assurance procedures performed on the data:
The stress-strain relation obtained from the testing machine is considered to assure linear elastic deformation of the specimen. Moreover, for the same reason it was verified that no permanent deformations of the specimen were present after the experiment.


DATA & FILE OVERVIEW

1. File List:
    - 20231116_displacements_raw.csv : Contains the measured displacements and the corresponding coordinates of the measurement points.
    - 20231116_displacements_interpolated.csv : Contains the measured displacements and the corresponding coordinates of the measurement points interpolated on a ragular grid as described above.
    - 20231116_geometry.png : Shows a figure of the specimen geometry.
    - 20231116_specimen.png : Shows the steel specimen with the applied speckle pattern.
    - 20231116_fem_element_incidences.csv : Contains the FEM element incidences used during the FEM-based calibration process described in the above referenced publication.
    - 20231116_fem_node_coordinate.csv : Contains the FEM node coordinates used during the FEM-based calibration process described in the above referenced publication. 
    - interpolationDIC2FE.m : Matlab-routine for the interpolation of the raw digital image correlation data onto a regular grid.
    - 20231116_mcmc_samples_fem.csv : Contains the samples obtained from a MCMC analysis using FEM and the emcee algorithm. For detailed information, see the above referenced publication.

2. Relationship between files, if important: 
    - "20231116_displacements_interpolated.csv" is generated from "20231116_displacements_raw.csv" using interpolationDIC2FE.m.

3. Explanation of the file name convention: 
	A. Structure: <date>_<description>.<file format>
	B. Examples: 20231116_displacements_raw.csv

4. File formats:
    - TXT-files
    - CSV-files
    - PNG-files
    - M-files (MATLAB files)


DATA-SPECIFIC INFORMATION FOR: [20231116_displacements_raw.csv]

1. Variable List:
    - x-coordinate [mm]
    - y-coordinate [mm]
    - x-displacement [mm] (axial displacement)
    - y-displacement [mm] (lateral displacement)

2. Units of measurement: Both the coordinates and the displacements are measured in millimeters (abbreviated by "mm").


DATA-SPECIFIC INFORMATION FOR: [20231116_displacements_interpolated.csv]

1. Variable List:
    - x-coordinate [mm]
    - y-coordinate [mm]
    - x-displacement [mm] (axial displacement)
    - y-displacement [mm] (lateral displacement)

2. Units of measurement: Both the coordinates and the displacements are measured in millimeters (abbreviated by "mm").


DATA-SPECIFIC INFORMATION FOR: [20231116_fem_element_incidences.csv]

1. Variable List:
    - element ID
    - node[1-8]

2. Units of measurement: The numerical values are all integers and have no unit of measurement.


DATA-SPECIFIC INFORMATION FOR: [20231116_fem_node_coordinate.csv]

1. Variable List:
    - element ID
    - x-coordinate [mm]
    - y-coordinate [mm]

2. Units of measurement: 
    - Element IDs are integers and have no unit of measurement. 
    - Coorinates are measured in millimeters (abbreviated by "mm").


DATA-SPECIFIC INFORMATION FOR: [20231116_mcmc_samples_fem.csv]

1. Variable List:
    - bulk modulus samples [N/mm^2]
    - shear modulus samples [N/mm^2]

2. Units of measurement: 
    - Both the bulk modulus samples and the shear modulus samples are measured in Newton per square millimeter (abbreviated by "N/mm^2"). 
