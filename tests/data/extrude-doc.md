# extrude

Extend a 2-dimensional sketch or individual segment of a sketch through a third dimension to create a new 3-dimensional volume or surface, or if extruded into an existing volume, cut into an existing solid.

Extend a 2-dimensional sketch or individual segment of a sketch through a third dimension to create a new 3-dimensional volume or surface, or if extruded into an existing volume, cut into an existing solid.

```kcl
extrude(
  @sketches: [Sketch | Face | TaggedFace | TaggedEdge | Edge | Segment | any; 1+],
  length?: number(Length),
  to?: Point3d | Axis3d | Plane | Edge | Face | Sketch | Solid | TaggedEdge | TaggedFace | any,
  symmetric?: bool,
  direction?: Point3d | Edge | TaggedEdge | Segment | any,
  bidirectionalLength?: number(Length),
  tagStart?: TagDecl,
  tagEnd?: TagDecl,
  draftAngle?: number(Angle),
  twistAngle?: number(Angle),
  twistAngleStep?: number(Angle),
  twistCenter?: Point2d,
  method?: string,
  hideSeams?: bool,
  bodyType?: string,
): [Solid; 1+]
```

You can provide more than one sketch to extrude, and they will all be
extruded in the same direction.
Plane-based sketches have no parent body, so each produces a separate body;
`MERGE` does not combine them. Use `union` when one body is intended.

When you sketch on a face of a solid, extruding extends or cuts into the
existing solid, meaning you don't need to union or subtract the volumes. You
can change this behavior by using the `method` parameter. See
[sketch on face](/docs/kcl-lang/sketch-on-face) for more details.

### Arguments

| Name | Type | Description | Required |
|----------|------|-------------|----------|
| `sketches` | [[`Sketch`](/docs/kcl-std/types/std-types-Sketch) or [`Face`](/docs/kcl-std/types/std-types-Face) or [`TaggedFace`](/docs/kcl-std/types/std-types-TaggedFace) or [`TaggedEdge`](/docs/kcl-std/types/std-types-TaggedEdge) or [`Edge`](/docs/kcl-std/types/std-types-Edge) or [`Segment`](/docs/kcl-std/types/std-types-Segment) or [`any`](/docs/kcl-std/types/std-types-any); 1+] | Which sketch or sketches should be extruded. Experimental face API: edge specifier objects (`{ sideFaces = [faceTag1, faceTag2], endFaces? = [...], index? }`) are not ready for generated or user-facing KCL yet; prefer existing sketch, face, tagged face, tagged edge, edge, or segment forms until point-and-click and migration support ships. | Yes |
| `length` | [`number(Length)`](/docs/kcl-std/types/std-types-number) | How far to extrude the given sketches. Incompatible with `to`. | No |
| `to` | [`Point3d`](/docs/kcl-std/types/std-types-Point3d) or [`Axis3d`](/docs/kcl-std/types/std-types-Axis3d) or [`Plane`](/docs/kcl-std/types/std-types-Plane) or [`Edge`](/docs/kcl-std/types/std-types-Edge) or [`Face`](/docs/kcl-std/types/std-types-Face) or [`Sketch`](/docs/kcl-std/types/std-types-Sketch) or [`Solid`](/docs/kcl-std/types/std-types-Solid) or [`TaggedEdge`](/docs/kcl-std/types/std-types-TaggedEdge) or [`TaggedFace`](/docs/kcl-std/types/std-types-TaggedFace) or [`any`](/docs/kcl-std/types/std-types-any) | Reference to extrude to. Incompatible with `length` and `twistAngle`. Experimental face API: edge specifier objects (`{ sideFaces = [faceTag1, faceTag2], endFaces? = [...], index? }`) are not ready for generated or user-facing KCL yet; prefer existing point, axis, plane, edge, face, sketch, solid, or tag forms until point-and-click and migration support ships. | No |
| `symmetric` | [`bool`](/docs/kcl-std/types/std-types-bool) | If true, the extrusion will happen symmetrically around the sketch. Otherwise, the extrusion will happen on only one side of the sketch. | No |
| `direction` | [`Point3d`](/docs/kcl-std/types/std-types-Point3d) or [`Edge`](/docs/kcl-std/types/std-types-Edge) or [`TaggedEdge`](/docs/kcl-std/types/std-types-TaggedEdge) or [`Segment`](/docs/kcl-std/types/std-types-Segment) or [`any`](/docs/kcl-std/types/std-types-any) | If specified, will extrude in this direction instead of the sketch plane normal. If an edge is being extruded, this defaults to halfway between the faces on either side of the edge. | No |
| `bidirectionalLength` | [`number(Length)`](/docs/kcl-std/types/std-types-number) | How far to extrude opposite the primary extrusion direction. Only used with `length`; incompatible with `symmetric`. | No |
| `tagStart` | [`TagDecl`](/docs/kcl-std/types/std-types-TagDecl) | A named tag for the face at the start of the extrusion, i.e. the original sketch. | No |
| `tagEnd` | [`TagDecl`](/docs/kcl-std/types/std-types-TagDecl) | A named tag for the face at the end of the extrusion, i.e. the new face created by extruding the original sketch. | No |
| `draftAngle` | [`number(Angle)`](/docs/kcl-std/types/std-types-number) | **Experimental.** Positive draft angle means the sketch gets smaller while extruding, i.e. inwards draft. Negative draft angle means the sketch gets bigger while extruding, i.e. outwards draft. Defaults to zero, i.e. no draft. | No |
| `twistAngle` | [`number(Angle)`](/docs/kcl-std/types/std-types-number) | If given, the sketch will be twisted around this angle while being extruded. Incompatible with `to`. | No |
| `twistAngleStep` | [`number(Angle)`](/docs/kcl-std/types/std-types-number) | The size of each intermediate angle as the sketch twists around. Must be between 4 and 90 degrees. Only used if `twistAngle` is given, defaults to 15 degrees. | No |
| `twistCenter` | [`Point2d`](/docs/kcl-std/types/std-types-Point2d) | The center around which the sketch will be twisted. Relative to the plane's origin. Only used if `twistAngle` is given, defaults to [0, 0] i.e. plane origin. | No |
| `method` | [`string`](/docs/kcl-std/types/std-types-string) | The method used during extrusion, either `NEW` or `MERGE`. `NEW` creates a new body. `MERGE` updates the parent body when extruding a face or a sketch on a face; it does not combine separate plane-based sketches. The default is `MERGE`. | No |
| `hideSeams` | [`bool`](/docs/kcl-std/types/std-types-bool) | Whether or not to hide the seams between the original and resulting object. Only used if a face is extruded and method = MERGE | No |
| `bodyType` | [`string`](/docs/kcl-std/types/std-types-string) | What type of body to produce (solid or surface). Defaults to "solid". | No |

### Returns

[[`Solid`](/docs/kcl-std/types/std-types-Solid); 1+]


### Examples

```kcl
example = startSketchOn(XZ)
  |> startProfile(at = [0, 0])
  |> line(end = [10, 0])
  |> arc(angleStart = 120deg, angleEnd = 0, radius = 5)
  |> line(end = [5, 0])
  |> line(end = [0, 10])
  |> bezierCurve(control1 = [-10, 0], control2 = [2, 10], end = [-5, 10])
  |> line(end = [-5, -2])
  |> close()
  |> extrude(length = 10)

```
