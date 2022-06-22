//========================================================================
// Drag and drop audio handling
//========================================================================

var FileSelect = document.getElementById("InputFile");
console.log(FileSelect);


// Add event listeners
FileSelect.addEventListener("change", fileSelectHandler, false);

function fileDragHover(e) {
    // prevent default behaviour
    e.preventDefault();
    e.stopPropagation();

}

function fileSelectHandler(e) {
    // handle file selecting
    var files = e.target.files || e.dataTransfer.files;
    fileDragHover(e);
    for (var i = 0, f; (f = files[i]); i++) {
        previewFile(f);
    }
}

//========================================================================
// Web page elements for functions to use
//========================================================================

var audioPreview = document.getElementById("audio-preview");
var audioDisplay = document.getElementById("audio-display");
var predResult = document.getElementById("pred-result");

//========================================================================
// Main button events
//========================================================================

function genderSubmitAudio() {
    // action for the submit button
    console.log("in src cua audio");
    console.log(audioDisplay);
    console.log(audioDisplay.src);

    if (!audioDisplay.src || !audioDisplay.src.startsWith("data")) {
        window.alert("Please select an audio before submit.");
        return;
    }

    audioDisplay.classList.add("loading");

    // call the predict function of the backend
    genderPredictAudio(audioDisplay.src);
}

function emotionSubmitAudio() {
    // action for the submit button
    console.log(audioDisplay.src);

    if (!audioDisplay.src || !audioDisplay.src.startsWith("data")) {
        window.alert("Please select an audio before submit.");
        return;
    }

    audioDisplay.classList.add("loading");

    // call the predict function of the backend


    var reader = new FileReader();
    // reader.readAsDataURL(blob);
    // reader.onloadend = () => {
    //     var base64data = reader.result;
    //     //log of base64data is "data:audio/ogg; codecs=opus;base64,GkX..."
    // }

    emotionPredictAudio(audioDisplay.src);
}

function previewFile(file) {
    // show the preview of the audio
    console.log(file.name);
    var fileName = encodeURI(file.name);

    var reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onloadend = () => {
        audioPreview.src = URL.createObjectURL(file);

        show(audioPreview);

        // reset
        predResult.innerHTML = "";
        audioDisplay.classList.remove("loading");

        displayAudio(reader.result, "audio-display");
    };
}


//========================================================================
// Helper functions
//========================================================================

function genderPredictAudio(audio) {
    console.log(audio)
    fetch("/gender/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(audio)
    })
        .then(resp => {
            if (resp.ok)
                resp.json().then(data => {
                    console.log(data);
                    displayResult(data);
                });
        })
        .catch(err => {
            console.log("An error occured", err.message);
            window.alert("Oops! Something went wrong.");
        });
}

function emotionPredictAudio(audio) {
    fetch("/emotion/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(audio)
    })
        .then(resp => {
            if (resp.ok)
                resp.json().then(data => {
                    displayResult(data);
                });
        })
        .catch(err => {
            console.log("An error occured", err.message);
            window.alert("Oops! Something went wrong.");
        });
}

function displayAudio(audio, id) {
    // display audio on given id <img> element
    let display = document.getElementById(id);
    display.src = audio;
    show(display);
}

function displayResult(data) {
    // display the result
    // audioDisplay.classList.remove("loading");
    predResult.innerHTML = data.result;
    show(predResult);
}

function hide(el) {
    // hide an element
    el.classList.add("hidden");
}

function show(el) {
    // show an element
    el.classList.remove("hidden");
}