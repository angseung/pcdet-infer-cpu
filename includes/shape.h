namespace vueron
{
    class Rectangle {
    public:
    Rectangle(int width, int height);
    Rectangle();
    ~Rectangle();

    int GetSize() const;

    private:
    int width_, height_;
    };
} //namespace vueron
