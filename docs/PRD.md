# HandForge — Product Requirements

## 1. Problem

In an avatar broadcast, the movement of the hands is judged by the viewer, who sees nothing but the avatar on screen. The performer is the person in front of the webcam whose hands the avatar follows. What the performer watches must be that same screen.

On that screen, both hands, the finger joints, and the arms must be seen to move naturally.

## 2. Goal

The two hands, finger joints, and arms of a VRM avatar follow the performer's, and whether that movement meets the acceptance criteria in §6 can be checked again at any time.

Deliverables:

1. the product: the tracker, which reads the webcam, and the avatar application, the Unity application that renders the avatar, as a standalone build,
2. instructions for installing and running the product,
3. a measurement procedure that repeats a run with the same steps under the same conditions,
4. a measurement record for each acceptance criterion, identifying what was measured by the commit, the Unity version and build options of the avatar application's build, and the runtime environment of the tracker, whose details §5.2 sets.

## 3. Scope

The performer can be anyone who chooses to run HandForge.

HandForge runs on one Windows machine with one webcam and one avatar.

The avatar must be a VRM avatar that defines all three humanoid bones of every finger.

What moves is both hands, the finger joints, and the arms. The arms are moved to follow the hands and are not tracked themselves.

## 4. Non-goals

Excluded from this release:

- remote tracking, meaning the tracker and the avatar application running on different machines
- output over the VMC protocol or OSC
- face or upper-body tracking
- driving more than one avatar at the same time
- operating systems other than Windows
- physical interaction between the avatar's hands and objects in the scene
- packaging for use in other Unity projects

## 5. Measurement

What is measured is the output of the avatar application.

A run is one continuous session of the product under one condition of §5.2, measured from its start to its end.

### 5.1 Metrics

Recorded on every run, whether or not it passes:

- **Observed items** — the observed items of §6, judged on the avatar application's output as the external camera records it beside the performer's hands.
- **Output frame time** — the interval between frames the avatar application presents, recorded by the avatar application itself. This recording is part of the product and is on by default. The 99th percentile and the longest frame time are recorded. The 0.1% low and the number of frames longer than 33.33 ms are recorded as diagnostics. 33.33 ms is two frame intervals of the 60 fps target (§6); a frame longer than that makes the broadcast repeat at least one frame. A stall is a frame the avatar application presents later than the 16.67 ms frame budget.
- **Latency** — the time from the performer's hand moving to that movement appearing, as output of the avatar application, on this machine's display. It includes the time the webcam takes to deliver the image, and is measured with an external camera. A latency event is one sudden hand movement the performer makes for this measurement.
- **Re-acquisition** — the time from the whole hand coming back into the webcam's view to the avatar's hand starting to follow it again, measured with an external camera. A re-acquisition event is one occasion on which the whole hand leaves the webcam's view, tracking is lost, and the hand comes back into view.
- **Diagnostics** — the internal interval from the tracker obtaining an image to the avatar application rendering it; the per-frame processing time and throughput the tracker reports about itself; the time spent in garbage collection in each frame; and, in a diagnostic run, the landmark values the tracker produces.

### 5.2 Conditions

A run is made under one of three conditions:

- **Acceptance run** — the product runs in the form the performer uses, with capture and encoding running on the same machine. Capture and encoding may write a local recording instead of going out live, with the encoder set as for a broadcast. The form the performer uses is the product installed and started the way HandForge instructs performers to, with its bundled settings unchanged; the avatar application is a standalone build without development instrumentation.
- **Comparison run** — the same form, without capture and encoding. It is used to tell whether a failure comes from the HandForge side or from the capture and encoding side.
- **Diagnostic run** — a development build with a profiler attached. It is used to look at garbage collection and at the causes of failures, and is not used for acceptance.

Running inside the Unity Editor is not a measurement condition, because the editor's own cost is included in the result.

The tracker and the avatar application run together on the same machine. The avatar application outputs at the target resolution of §6.

Runs that are compared with each other use the same avatar, and the record identifies it.

An acceptance run lasts 60 minutes. A run that checks a change during development lasts at least 30 minutes.

The measurement record lists every input that can change the result. The measurement procedure (§2) sets how each is written down.

- the machine: its model and configuration, including its display
- the webcam and its settings
- the capture and encoding software and its settings
- the external camera and its frame rate, which is the resolution of the latency and re-acquisition measurements
- the tracker's settings and interpreter version
- the hand-landmark model file
- the avatar
- the performer, the lighting, and the background behind the performer as the webcam sees it, described rather than fixed

An acceptance result holds only for the conditions recorded with it.

### 5.3 Procedure

A run is carried out by the performer alone.

During a run, the performer follows a fixed scenario while acting as in a broadcast. The scenario contains, for each observed item of §6, actions that show it, and it contains latency events and re-acquisition events, with the hand leaving the view both briefly and for longer. These actions and events recur from the beginning of the run to its end, so a change that builds up during the run also shows.

The external camera films the whole run.

A run counts as a failure when the tracker or the avatar application stops, or when the frame-time record or the tracker's report breaks off because of the product; how the avatar's hands look while the tracker is stopped is judged by §6. A run whose recording or records break off for a reason outside the product is not judged and is run again.

The measurement procedure sets the content of the scenario, the number of events, the camera setup, the rules for judging, how a moment in the records is matched to the same moment in the recording, and what is kept from a run.

## 6. Acceptance criteria

This release targets a broadcast at 1920×1080 and 60 fps. A 60 fps broadcast takes a new frame every 16.67 ms, and a frame the avatar application presents later than that makes the broadcast repeat a frame, so the frame budget is 16.67 ms. This release also targets less than 100 ms from the performer's hand moving to that movement appearing on this machine's display.

These targets are set by this release. They are not a claim that a viewer cannot notice a stall or a delay at those values.

The natural movement required by §1 is judged by the criteria below. Every valid acceptance run must meet all of them.

### Observed items

A person watches the recording of the whole run from start to end, as a viewer would, compares the avatar with the performer's real hands, and notes every moment an item below fails. An item that fails once fails the run. The scenario makes each item appear at least once.

1. Each finger's three joints bend following the performer's finger.
2. Both hands are tracked at the same time, and each follows the same hand of the performer for the whole run: the avatar's left hand follows the performer's left hand, and the right follows the right.
3. The wrist stays attached to the arm, and the arm follows the hand.
4. When the hand stops, the avatar's hand stops.
5. While the hand moves, the avatar's hand moves smoothly, without shaking or stepping that the performer's hand does not show.
6. A hand that is not being tracked — before it is first found, while covered, out of the webcam's view, or while the tracker's data has stopped — keeps its pose for a moment, then moves to a resting pose with the arm lowered, and follows again once it is tracked. The moment and the longer case are judged by the scenario's brief and longer departures.
7. The avatar's hand never jumps to another place in one step, including when a hand is lost and when it is found again.

### Measured values

1. The 99th percentile of output frame time is at most 16.67 ms. A mean is not used, because it hides frames over budget; 100% is not used, because a single outlying frame would then fail the run.
2. Every latency event is under 100 ms.
3. Every re-acquisition event is under 100 ms.
4. The longest frame time plus the largest latency is under 100 ms. This is a conservative bound, not a measured value: it covers a stall at a moment no latency event sampled, and counts a stall twice when one did.

Events are judged all together rather than by percentile, because a viewer sees each movement on its own and a performer produces few events in a run.

## 7. Stop rule

Implementation and tuning for this release close when either condition holds:

1. there is a record of an acceptance run that meets every criterion of §6, or
2. a criterion is not met, and the comparison run and the diagnostics identify the cause.

Closing without either is not permitted.

When the cause is identified as the hand-landmark model, the release closes under condition 2, and the record states that the landmark values were already wrong at the failing moment and whether processing after the model could have prevented it. If the model makes a criterion unreachable, this document is revised.

A change to the build or to a recorded input makes a new configuration, and the criteria are judged again. This is normal operation, not a reopening of closed work.
