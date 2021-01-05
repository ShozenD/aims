var scores = {"KIME":"","KEANA":"","SHIMI":"","TARUMI":"","TOUMEI":"","MERANIN":"","KOJIWA":""};

const spawn = require("child_process").spawn;
const pythonProcess = spawn('python',["ev_skin.py", "2000.jpg",]);
pythonProcess.stdout.on('data', (data) => {
    const str = data.toString();
    const replace = str.replace(/\r\n/g, "");
    const result = replace.split(' ');
    var i = 0;
    for( let label in scores ){
      scores[label] = result[i];
      i++;
    }
    console.log(scores);
});
