document.addEventListener('DOMContentLoaded', function() {
    var navLinks = document.querySelectorAll('div a');
  
    for (var i = 0; i < navLinks.length; i++) {
      navLinks[i].addEventListener('click', scrollToSection);
    }
  
    function scrollToSection(event) {
      event.preventDefault();
  
      var targetId = this.getAttribute('href');
      var targetElement = document.querySelector(targetId);
      targetElement.scrollIntoView({
        behavior: 'smooth'
      });
    }
});


function saveContent() {
  const content = document.getElementById('editableContent').innerText;
  localStorage.setItem('savedContent', content);
  alert('Content saved!');
}
window.onload = function() {
  const savedContent = localStorage.getItem('savedContent');
  if (savedContent) {
    document.getElementById('editableContent').innerText = savedContent;
  }
};

function addCrossLine(cell) {
  cell.classList.toggle('cross-line');
}

// To-do-list
document.addEventListener('DOMContentLoaded', function(){

let button = document.getElementById('add');
let todolist = document.getElementById('todoList');
let input = document.getElementById('input');

// Local Storage
let todos = [];
window.onload = ()=>{
  todos = JSON.parse(localStorage.getItem('todos')) || []
  todos.forEach(todo =>addtodo(todo))  
}

button.addEventListener('click',()=>{
  todos.push(input.value)
  localStorage.setItem('todos',JSON.stringify(todos))
  console.log(todos)
  addtodo(input.value)
  input.value = ''
});

function addtodo(todo) {
  let para = document.createElement('p');
  para.innerText = todo;
  todolist.appendChild(para)

  para.addEventListener('click',()=>{
    para.style.textDecoration = 'line-through'
    remove(todo)
  })
  para.addEventListener('dblclick',()=>{
    todolist.removeChild(para)
    remove(todo)
  })
}

function remove(todo) {
 let index = todos.indexOf(todo)
 if (index > -1) {
  todos.splice(index,1)
 }
 localStorage.setItem('todos',JSON.stringify(todos))
}

});