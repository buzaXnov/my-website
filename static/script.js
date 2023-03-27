var canvas = document.getElementById('canvas');
var ctx = canvas.getContext('2d');
var drawing = false;
var strokes = [];

canvas.addEventListener('mousedown', function (e) {
    if (!drawing) {
        // Clear the canvas when the mouse if clicked for a new symbol to be drawn
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.beginPath();
        strokes = [];
    }
    drawing = true;
    strokes.push([e.offsetX, e.offsetY]);
});

canvas.addEventListener('mousemove', function (e) {
    if (drawing) {
        ctx.beginPath();
        ctx.moveTo(strokes[strokes.length - 1][0], strokes[strokes.length - 1][1]);
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
        strokes.push([e.offsetX, e.offsetY]);
    }
});

canvas.addEventListener('mouseup', function (e) {
    drawing = false;
    // Send the strokes data to the server for recognition
    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/greek-symbols');
    xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
    xhr.onload = function () {
        if (xhr.status === 200) {
            alert(xhr.responseText);
        } else {
            alert('Error: ' + xhr.statusText);
        }
    };
    xhr.send('strokes=' + JSON.stringify(strokes));
    strokes = [];
});

// Commented so that every page can load the particles.js script and have it as the background. 
// particlesJS.load('particles-js', 'particles.json', function () {
//     console.log('loaded successfully');
// });