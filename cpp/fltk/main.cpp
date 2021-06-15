#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Button.H>

int main(int argc, char **argv)
{
    Fl_Window *window = new Fl_Window(340, 180);

    Fl_Group *group = new Fl_Group(0, 0, 300, 100, "No Resizing");

    Fl_Box *box = new Fl_Box(20, 40, 300, 100, "Hello, World!");

    Fl_Button *button = new Fl_Button(20, 20, 100, 100, "label");

    box->box(FL_UP_BOX);
    box->labelfont(FL_BOLD + FL_ITALIC);
    box->labelsize(36);
    box->labeltype(FL_SHADOW_LABEL);
    group->resizable(group);
    group->end();

    window->resizable(window);

    window->end();
    window->show(argc, argv);

    return Fl::run();
}